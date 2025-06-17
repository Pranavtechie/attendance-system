from peewee import CharField, DateTimeField, Model, SqliteDatabase

db = SqliteDatabase('kcc.db')

class Cadet(Model):
    uniqueId = CharField(primary_key=True)
    name = CharField()
    admissionNumber = CharField()
    roomId = CharField()
    pictureFileName = CharField()
    syncedAt = DateTimeField()

    class Meta:
        database = db

class Room(Model):
    roomId = CharField(primary_key=True)
    roomName = CharField()
    syncedAt = DateTimeField()

    class Meta:
        database = db
    

class CadetAttendance(Model):
    uniqueId = CharField()
    attendanceTimeStamp = DateTimeField()
    sessionId = CharField()
    syncedAt = DateTimeField()

    class Meta:
        database = db  

class Session(Model):
    sessionId = CharField(primary_key=True)
    sessionTimestamp = DateTimeField()
    syncedAt = DateTimeField()

    class Meta:
        database = db

if __name__ == "__main__":
    db.connect()
    db.create_tables([Cadet, Room, CadetAttendance, Session], safe=True)
    db.close()