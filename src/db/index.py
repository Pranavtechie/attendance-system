from peewee import Model, CharField, SqliteDatabase, IntegerField, DateTimeField

db = SqliteDatabase('people.db')


class Cadet(Model):
    uniqueId = CharField(primary_key=True)
    name = CharField()
    admissionNumber = CharField()
    roomName  = CharField()

    class Meta:
        database = db

class Room(Model):
    roomName = CharField(primary_key=True)
    roomCapacity = IntegerField()

    class Meta:
        database = db
    

class SyncValidator(Model):
    syncHash = CharField()
    syncTimestamp = DateTimeField()

    class Meta:
        database = db

class CadetAttendance(Model):
    uniqueId = CharField()
    attendanceTimeStamp = DateTimeField()
    session = CharField()

    class Meta:
        database = db   

if __name__ == "__main__":
    print('something happened')
    db.connect()
    db.create_tables([Cadet, Room, SyncValidator, CadetAttendance], safe=True)
    db.close()