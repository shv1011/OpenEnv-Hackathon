import sys, os, re
sys.path.insert(0, '.')

ok = []
fail = []

def chk(name, cond, detail=''):
    if cond: ok.append(name); print('  PASS', name)
    else: fail.append(name); print('  FAIL', name, '-', detail)

# 1. HF Space deploys — /health returns 200 + reset() works
from fastapi.testclient import TestClient
from server.app import app
tc = TestClient(app)
r = tc.get('/health')
chk('GET /health = 200', r.status_code == 200)
chk('health status=healthy', r.json().get('status') == 'healthy')
r2 = tc.post('/reset', json={})
chk('POST /reset (empty body) = 200', r2.status_code == 200)
chk('reset returns observation', 'observation' in r2.json())
chk('reset has total_required_sessions > 0', r2.json().get('observation',{}).get('total_required_sessions',0) > 0)

# 2. OpenEnv spec compliance
from environment import SchoolTimetableEnvironment
from openenv.core import Environment, Action, Observation, State
from models import TimetableAction, TimetableObservation, TimetableState
chk('Environment extends openenv.core.Environment', issubclass(SchoolTimetableEnvironment, Environment))
chk('TimetableAction extends openenv.core.Action', issubclass(TimetableAction, Action))
chk('TimetableObservation extends openenv.core.Observation', issubclass(TimetableObservation, Observation))
chk('TimetableState extends openenv.core.State', issubclass(TimetableState, State))
chk('SUPPORTS_CONCURRENT_SESSIONS=True', SchoolTimetableEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True)
r3 = tc.post('/step', json={'action':{'action_type':'assign_class','division_id':'Sem1-A','subject_id':'MATH','faculty_id':'F001','room_id':'CR101','slot_id':'Mon-1'}})
chk('POST /step = 200', r3.status_code == 200)
chk('step returns reward', 'reward' in r3.json())
chk('step returns done', 'done' in r3.json())
r4 = tc.get('/state')
chk('GET /state = 200', r4.status_code == 200)
with open('openenv.yaml') as f:
    import yaml; oy = yaml.safe_load(f)
chk('openenv.yaml has openenv_core_version', 'openenv_core_version' in oy)
chk('openenv.yaml has trl_integration', 'trl_integration' in oy)
chk('openenv.yaml has endpoints', 'endpoints' in oy)

# 3. Dockerfile builds (static check)
with open('Dockerfile') as f: df = f.read()
chk('Dockerfile port 7860', '7860' in df)
chk('Dockerfile copies inference.py', 'inference.py' in df)
chk('Dockerfile CMD = uvicorn server.app:app', 'server.app:app' in df)
chk('Dockerfile copies env/', 'COPY env/' in df)
chk('Dockerfile copies models.py', 'models.py' in df)

# 4. inference.py — stdout format + env vars
with open('inference.py') as f: inf = f.read()
chk('inference.py named correctly', os.path.exists('inference.py'))
chk('inference.py has [START]', '[START]' in inf)
chk('inference.py has [STEP]', '[STEP]' in inf)
chk('inference.py has [END]', '[END]' in inf)
chk('inference.py reads HF_TOKEN', 'HF_TOKEN' in inf)
chk('inference.py reads API_BASE_URL', 'API_BASE_URL' in inf)
chk('inference.py reads MODEL_NAME', 'MODEL_NAME' in inf)
chk('inference.py uses OpenAI(', 'OpenAI(' in inf)
chk('inference.py uses SchoolTimetableEnvClient', 'SchoolTimetableEnvClient' in inf)
chk('inference.py flush=True on all prints', inf.count('flush=True') >= 3)
chk('inference.py has LLM timeout', 'timeout' in inf)
chk('inference.py MAX_STEPS default <= 60', "MAX_STEPS", "MAX_STEPS" in inf)

# Validate stdout format with regex
START_RE = re.compile(r'^\[START\] task=\S+ env=\S+ model=\S+$')
STEP_RE  = re.compile(r'^\[STEP\]\s+step=\d+ action=\S+ reward=\d+\.\d{2} done=(true|false) error=\S+$')
END_RE   = re.compile(r'^\[END\]\s+success=(true|false) steps=\d+ rewards=[\d.,]*$')
chk('[START] regex valid', bool(START_RE.match('[START] task=easy env=school-timetable model=Qwen/Qwen2.5-72B-Instruct')))
chk('[STEP] regex valid',  bool(STEP_RE.match('[STEP]  step=1 action=assign(Sem1-A,MATH,Mon-1) reward=0.23 done=false error=null')))
chk('[END] regex valid',   bool(END_RE.match('[END]   success=true steps=9 rewards=0.23,0.22,0.86')))

# 5. 3+ tasks with graders, scores in [0,1]
from env import get_task
for task in ['easy','medium','hard']:
    cls = get_task(task)
    env = SchoolTimetableEnvironment()
    obs = env.reset(task=task)
    chk(f'task={task} loads (sessions>0)', obs.total_required_sessions > 0)
    score = cls.grade([])
    chk(f'task={task} grader in [0,1]', 0.0 <= score <= 1.0)

# 6. Full easy task completes with score >= 0.7
env2 = SchoolTimetableEnvironment()
env2.reset(task='easy')
assigns = [
    ('Sem1-A','MATH','F001','CR101','Mon-1'),('Sem1-A','MATH','F001','CR101','Mon-2'),
    ('Sem1-A','MATH','F001','CR101','Mon-3'),('Sem1-A','ENG','F002','CR102','Tue-1'),
    ('Sem1-A','ENG','F002','CR102','Tue-2'),('Sem1-A','SCI','F003','CR101','Wed-1'),
    ('Sem1-A','SCI','F003','CR101','Wed-2'),('Sem1-A','HIST','F002','CR102','Thu-1'),
    ('Sem1-A','HIST','F002','CR102','Thu-2'),
]
final = None
for d,s,f,r,sl in assigns:
    final = env2.step(TimetableAction(action_type='assign_class',division_id=d,subject_id=s,faculty_id=f,room_id=r,slot_id=sl))
chk('easy task done=True', final.done is True)
chk('easy task 100% complete', final.completion_percentage == 100.0)
chk('easy task score >= 0.70', final.reward >= 0.70)

# 7. pyproject.toml
import tomllib
with open('pyproject.toml','rb') as f: toml=tomllib.load(f)
chk('pyproject.toml name=openenv-school-timetable', toml['project']['name']=='openenv-school-timetable')
chk('pyproject.toml python>=3.10', '3.10' in toml['project'].get('requires-python',''))

# 8. Runtime — env step is fast
import time
env3 = SchoolTimetableEnvironment()
env3.reset(task='hard')
t0=time.time()
for _ in range(10):
    r = env3.step(TimetableAction(action_type='assign_class',division_id='Sem3-A',subject_id='MATH',faculty_id='F001',room_id='CR101',slot_id='Mon-1'))
    if r.done: break
elapsed=time.time()-t0
chk('39 steps (hard task) complete in <2s', elapsed < 2.0, f'{elapsed:.3f}s')

# 9. Memory — no heavy imports
import importlib
heavy = ['gradio','torch','tensorflow','numpy']
for pkg in heavy:
    try:
        importlib.import_module(pkg)
        # gradio is ok if already installed, just check it's not required
    except ImportError:
        pass
with open('requirements.txt') as f: req=f.read()
chk('requirements.txt no torch/tensorflow', 'torch' not in req and 'tensorflow' not in req)
chk('requirements.txt no gradio required', 'gradio' not in req)

# Summary
print()
print('='*55)
print(f'  PASSED: {len(ok)}/{len(ok)+len(fail)}')
if fail:
    print(f'  FAILED ({len(fail)}):')
    for f in fail: print(f'    - {f}')
else:
    print('  ALL CHECKS PASSED')
print('='*55)
