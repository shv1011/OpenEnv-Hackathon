import re, sys
sys.path.insert(0, '.')

START_RE = re.compile(r'^\[START\] task=\S+ env=\S+ model=\S+$')
STEP_RE  = re.compile(r'^\[STEP\]\s+step=\d+ action=\S+ reward=\d+\.\d{2} done=(true|false) error=\S+$')
END_RE   = re.compile(r'^\[END\]\s+success=(true|false) steps=\d+ score=\d+\.\d{2} rewards=[\d.,]*$')

lines = [
    "[START] task=easy env=school-timetable model=Qwen/Qwen2.5-72B-Instruct",
    "[STEP]  step=1 action=assign(Sem1-A,MATH,Mon-1) reward=0.23 done=false error=null",
    "[STEP]  step=9 action=assign(Sem1-A,HIST,Thu-2) reward=0.86 done=true error=null",
    "[END]   success=true steps=9 score=0.86 rewards=0.23,0.23,0.21,0.23,0.22,0.25,0.23,0.30,0.86",
]

assert START_RE.match(lines[0]), "BAD [START]"
assert STEP_RE.match(lines[1]),  "BAD [STEP]"
assert STEP_RE.match(lines[2]),  "BAD [STEP] done=true"
assert END_RE.match(lines[3]),   "BAD [END]"

for l in lines: print(l)
print()
print("FORMAT OK - all lines match new spec")

# Also verify inference.py has score= in [END]
with open('inference.py') as f: inf = f.read()
assert 'score={score:.2f}' in inf, "inference.py missing score= in [END]"
assert 'score = 0.0' in inf or 'score = compute_score' in inf or 'score=0.0' in inf, "inference.py missing score init"
assert 'compute_score' in inf, "inference.py missing grader call"
print("inference.py score field: OK")
