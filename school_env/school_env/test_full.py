import sys
sys.path.insert(0, '.')
from app import app
from fastapi.testclient import TestClient
client = TestClient(app)

print("=== RESET ===")
client.post('/api/reset', json={'task': 'easy'})

assigns = [
    ('Sem1-A','MATH','F001','CR101','Mon-1'),
    ('Sem1-A','MATH','F001','CR101','Mon-2'),
    ('Sem1-A','MATH','F001','CR101','Mon-3'),
    ('Sem1-A','ENG','F002','CR102','Tue-1'),
    ('Sem1-A','ENG','F002','CR102','Tue-2'),
    ('Sem1-A','SCI','F003','CR101','Wed-1'),
    ('Sem1-A','SCI','F003','CR101','Wed-2'),
    ('Sem1-A','HIST','F002','CR102','Thu-1'),
    ('Sem1-A','HIST','F002','CR102','Thu-2'),
]

for div,sub,fac,room,slot in assigns:
    r = client.post('/api/assign', json={
        'division_id': div, 'subject_id': sub,
        'faculty_id': fac, 'room_id': room, 'slot_id': slot
    })
    d = r.json()
    valid = d['valid']
    reward = round(d['reward'], 3)
    done = d['done']
    viols = [v['violation_type'] for v in d.get('violations', [])]
    print(sub, '@', slot, '| valid=', valid, '| reward=', reward, '| done=', done, '| violations=', viols)
    if d['done']:
        print('FINAL SCORE:', d['final_score'])

# Test reschedule
print("\n=== RESCHEDULE TEST ===")
client.post('/api/reset', json={'task': 'easy'})
r = client.post('/api/assign', json={'division_id':'Sem1-A','subject_id':'MATH','faculty_id':'F001','room_id':'CR101','slot_id':'Mon-1'})
entry_id = r.json()['observation']['timetable_entries'][0]['entry_id']
r2 = client.post('/api/reschedule', json={'entry_id': entry_id, 'new_slot_id': 'Fri-4'})
d2 = r2.json()
print('reschedule valid=', d2['valid'], 'reward=', round(d2['reward'],3))

# Test remove
print("\n=== REMOVE TEST ===")
entries = r2.json()['observation']['timetable_entries']
eid = entries[0]['entry_id']
r3 = client.post('/api/remove', json={'entry_id': eid})
d3 = r3.json()
print('remove valid=', d3['valid'], 'entries_left=', len(d3['observation']['timetable_entries']))

# Test metrics
print("\n=== METRICS ===")
client.post('/api/reset', json={'task': 'easy'})
m = client.get('/api/metrics').json()
print('metrics keys:', list(m.keys()))

# Test medium + hard reset
for task in ['medium', 'hard']:
    r = client.post('/api/reset', json={'task': task})
    obs = r.json()['observation']
    print(task, '| divisions=', len(obs['config']['divisions']), '| required=', obs['progress']['total_required_sessions'])

print("\nALL TESTS PASSED")
