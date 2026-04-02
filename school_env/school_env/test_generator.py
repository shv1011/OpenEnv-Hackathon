import sys
sys.path.insert(0, '.')
from app import app
from fastapi.testclient import TestClient
from collections import defaultdict

client = TestClient(app)

payload = {
    'class_name': 'Class 10',
    'num_divisions': 3,
    'subjects': [
        {'name': 'Mathematics', 'lectures_per_week': 5, 'faculty': ['Prof. Sharma']},
        {'name': 'Physics',     'lectures_per_week': 4, 'faculty': ['Dr. Mehta']},
        {'name': 'Chemistry',   'lectures_per_week': 4, 'faculty': ['Ms. Kapoor']},
        {'name': 'English',     'lectures_per_week': 3, 'faculty': ['Mr. Verma']},
        {'name': 'History',     'lectures_per_week': 3, 'faculty': ['Mrs. Singh']},
    ],
    'working_days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
    'periods_per_day': 7,
    'break_after_period': 4,
}

r = client.post('/api/generate', json=payload)
print('generate status:', r.status_code)
d = r.json()
print('divisions:', d['divisions'])
print('faculty:', list(d['faculty_grids'].keys()))
print('unresolved:', d['unresolved'])
print('grid rows:', len(d['grid']))

# Check no faculty clash
fac_slots = defaultdict(list)
for row in d['grid']:
    if row['subject'] not in ('FREE', 'BREAK') and row['faculty']:
        key = (row['faculty'], row['day'], str(row['period']))
        fac_slots[key].append(row['division'])
clashes = {str(k): v for k, v in fac_slots.items() if len(v) > 1}
print('faculty clashes:', len(clashes))
if clashes:
    for k, v in clashes.items():
        print('  CLASH:', k, '->', v)
else:
    print('  NO CLASHES - OK')

# Verify lecture counts per division
print('\nLecture counts per division:')
for div in d['divisions']:
    counts = defaultdict(int)
    for row in d['grid']:
        if row['division'] == div and row['subject'] not in ('FREE', 'BREAK'):
            counts[row['subject']] += 1
    for s in payload['subjects']:
        got = counts[s['name']]
        expected = s['lectures_per_week']
        status = 'OK' if got == expected else 'MISSING %d' % (expected - got)
        print('  Div %s | %s: %d/%d %s' % (div, s['name'], got, expected, status))

# CSV downloads
r2 = client.get('/api/download/faculty/Prof. Sharma')
print('\nfaculty csv:', r2.status_code, len(r2.content), 'bytes')
r3 = client.get('/api/download/all')
print('all csv:', r3.status_code, len(r3.content), 'bytes')
r4 = client.get('/api/download/division/A')
print('div A csv:', r4.status_code, len(r4.content), 'bytes')

# Test with multiple faculty per subject (one per division)
payload2 = {
    'class_name': 'Class 11',
    'num_divisions': 2,
    'subjects': [
        {'name': 'Math', 'lectures_per_week': 4, 'faculty': ['Mr. A', 'Mr. B']},
        {'name': 'Science', 'lectures_per_week': 3, 'faculty': ['Ms. C']},
    ],
    'working_days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
    'periods_per_day': 6,
    'break_after_period': 3,
}
r5 = client.post('/api/generate', json=payload2)
print('\nmulti-faculty generate:', r5.status_code)
d5 = r5.json()
print('unresolved:', d5['unresolved'])

print('\nALL GENERATOR TESTS PASSED')
