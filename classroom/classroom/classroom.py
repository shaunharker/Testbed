import torch
import asyncio


class Classroom:
    def __init__(self):
        self.students = {}

    def enroll(self, student):
        async def _train(student, students):
            try:
                while True:
                    dt = lambda: max(student.time - colleague.time for colleague in students)
                    while dt() > 0.0:
                        await asyncio.sleep(dt())
                    student.study()
                    if len(students) == 1:
                        await asyncio.sleep(1e-4)
            except Exception as e:
                print(f'classroom: exception in _train:')
                print(f'classroom:     e       = {e}')
                print(f'classroom:     type(e) = {type(e)}')
                raise e
            return None
        self.students[student] = asyncio.create_task(_train(student, self.students))

    def graduate(self, student):
        self.students[student].cancel()
        del self.students[student]
        return student
