import MySQLdb as mdb
import MySQLdb.cursors

import configuration as cfg

# set up database connection
mysql_data = cfg.data['database']
con = mdb.connect(user=mysql_data['user'], passwd=mysql_data['passwd'],
                  db=mysql_data['db'], charset='utf8', host=mysql_data['host'],
                  cursorclass=MySQLdb.cursors.DictCursor)


def get_last_experiment_id():
    with con:
        cur = con.cursor()
        query = "SELECT MAX(id) as id FROM BG_experiment_run"
        cur.execute(query)
        return cur.fetchone()["id"]


def load_patients():
    """
    Retrieve the patientIDs from database
    """
    patient_ids = list()
    with con:
        cur = con.cursor()
        query = "SELECT id FROM BG_Patient " \
                "WHERE id not in (9) "
        cur.execute(query)
        rows = cur.fetchall()
        if not rows:
            return
        for row in rows:
            patient_ids.append(row['id'])
        assert len(patient_ids) != 0
        return patient_ids


def loadPatientData():
    pass


curExpId = get_last_experiment_id() + 1
