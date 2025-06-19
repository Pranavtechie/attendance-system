from peewee import CharField, DateTimeField, Model, SqliteDatabase

db = SqliteDatabase("kcc.db")


class Person(Model):
    uniqueId = CharField(primary_key=True)
    name = CharField(null=False)
    admissionNumber = CharField()
    roomId = CharField()
    pictureFileName = CharField(null=False)
    personType = CharField(null=False)  # Cadet, Employee
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
    personId = CharField()
    attendanceTimeStamp = DateTimeField()
    sessionId = CharField()
    syncedAt = DateTimeField()

    class Meta:
        database = db


class Session(Model):
    sessionId = CharField(primary_key=True)
    startAt = DateTimeField()
    endedAt = DateTimeField()
    syncedAt = DateTimeField()

    class Meta:
        database = db


if __name__ == "__main__":
    db.connect()
    db.create_tables([Person, Room, CadetAttendance, Session], safe=True)
    db.close()
