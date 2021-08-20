import torch
import asyncio


class Classroom:
    def __init__(self):
        self.students = {}
        self.gun = None # but what if there are bears?

    def enroll(self, student):
        async def _train(student, students):
            try:
                time = lambda: asyncio.get_event_loop().time()
                while True:
                    dt = lambda: max(student.time - colleague.time for colleague in students)
                    while dt() > 0.0:
                        #print(f"{student.time}, {seat[0].time}, wait time = {wait_time}")
                        await asyncio.sleep(dt())
                    student.study()
                    if len(students) == 1:
                        await asyncio.sleep(1e-4)
            except Exception as e:
                print(e)
                raise e
            return None
        self.students[student] = asyncio.create_task(_train(student, self.students))

    def graduate(self, student):
        self.students[student].cancel()
        del self.students[student]
        return student
