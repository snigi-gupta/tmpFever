import yaml
from orator import DatabaseManager

f = open('db.yml', 'r')
KEYS = yaml.safe_load(f.read())
f.close()

DATABASES = {
    'mysql': {
        'driver': 'mysql',
        'host': KEYS['host'],
        'database': 'backup',
        'user': KEYS['user'],
        'password': KEYS['password'],
        'prefix': ''
    }
}



db = DatabaseManager(DATABASES)
