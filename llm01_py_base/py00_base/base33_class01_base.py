from __future__ import annotations

from datetime import datetime
import json

import pytest

## 3. 阶段 1：类与对象基础
### 3.1 练习一：学生对象
class Student:

    school_name = "Unknown School"
    student_count = 0

    def __init__(self, name: str, student_id: int, scores: dict, created_at: datetime):
        # - 姓名不能为空；
        if name is None or name == "":
            raise ValueError("Student name cannot be None")
        # - 学号不能为空；
        if student_id is None:
            raise ValueError("Student student_id cannot be None")
        # -
        if scores is not None:
            for course, score in scores.items():
                self._validate_score(course, score)

        self.name = name
        self.student_id = student_id
        self.scores = scores
        self.created_at = created_at

        Student.student_count += 1

    def _validate_score(self, course, score):
        """ 校验成绩是否合法 """
        # - 课程名不能为空；
        if course is None:
            raise ValueError("course cannot be None")
        # - 成绩必须在 0 到 100 之间；
        if 0 > score or score > 100:
            raise ValueError("Score must be positive, range in [0, 100]")

    def add_score(self, course, score) -> dict:
        """ 增加或修改成绩 """
        self._validate_score(course, score)

        self.scores.update({course: score})
        return self.scores

    def get_score(self, course) -> float:
        """ 获取指定课程成绩 """
        # - 查询不存在的课程时要有明确行为，不允许静默返回错误数据。
        keys = self.scores.keys()
        for key in keys:
            if course == key:
                return self.scores[key]
        raise KeyError("Course not found")

    def average_score(self) -> float:
        """ 计算平均分 """
        count = 0
        sum = 0
        for course, score in self.scores.items():
            count += 1
            sum += score
        if count == 0:
            return 0
        return sum / count

    def is_passed(self) -> bool:
        """ 判断平均分是否及格 """
        return self.average_score() >= 60

    def to_dict(self) -> dict:
        """ 将当前对象转化为普通字典 """
        return self.__dict__

    def to_json(self) -> str:
        """ 将当前对象转化为json """
        return json.dumps(self.to_dict())

    def __str__(self):
        """ 输出适合用户阅读的信息 """
        return (
            f"Student("
            f"name={self.name}, "
            f"student_id={self.student_id}, "
            f"scores={json.dumps(self.scores)}, "
            f"created_at={self.created_at}"
            f")"
        )

    def __repr__(self):
        """ 输出适合调试的信息 """
        return str(self)

    @classmethod
    def from_dict(cls, data: dict) -> Student:
        return Student(data.get("name"), data.get("student_id"), data.get("scores"), datetime.now())

    pass

def test_student_class():
    """ ### 3.1 练习一：学生对象 """

    print("test_student_class()")

    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 90,
            "English": 80,
            "Python": 95,
        },
        created_at=datetime(2026, 9, 30, 10, 0, 0),
    )

    assert student.name == "Tom"
    assert student.student_id == 10001

    assert student.scores == {
        "Math": 90,
        "English": 80,
        "Python": 95,
    }

    assert student.created_at == datetime(2026, 9, 30, 10, 0, 0)

def test_student_name_cannot_be_none():
    with pytest.raises(ValueError, match="Student name cannot be None"):
        Student(
            name=None,
            student_id=10001,
            scores={},
            created_at=datetime.now(),
        )

def test_student_id_cannot_be_none():
    with pytest.raises(
        ValueError,
        match="Student student_id cannot be None"
    ):
        Student(
            name="Tom",
            student_id=None,
            scores={},
            created_at=datetime.now(),
        )

def test_valid_scores():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 0,
            "English": 60,
            "Python": 100,
        },
        created_at=datetime.now(),
    )

    assert student.scores["Math"] == 0
    assert student.scores["English"] == 60
    assert student.scores["Python"] == 100

def test_score_cannot_be_less_than_zero():
    with pytest.raises(
        ValueError,
        match="Score must be positive, range in \\[0, 100\\]"
    ):
        Student(
            name="Tom",
            student_id=10001,
            scores={
                "Math": -1
            },
            created_at=datetime.now(),
        )

def test_score_cannot_be_greater_than_100():
    with pytest.raises(
        ValueError,
        match="Score must be positive, range in \\[0, 100\\]"
    ):
        Student(
            name="Tom",
            student_id=10001,
            scores={
                "Math": 101
            },
            created_at=datetime.now(),
        )

def test_course_cannot_be_none():
    with pytest.raises(ValueError, match="course cannot be None"):
        Student(
            name="Tom",
            student_id=10001,
            scores={
                None: 90
            },
            created_at=datetime.now(),
        )

def test_add_score():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 90
        },
        created_at=datetime.now(),
    )

    result = student.add_score("English", 80)

    assert result == {
        "Math": 90,
        "English": 80,
    }

    assert student.scores["English"] == 80

def test_update_score():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 90
        },
        created_at=datetime.now(),
    )

    result = student.add_score("Math", 95)

    assert result == {
        "Math": 95
    }

    assert student.scores["Math"] == 95

def test_add_invalid_score():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 90
        },
        created_at=datetime.now(),
    )

    with pytest.raises(ValueError):
        student.add_score("English", 101)

def test_get_score():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 90,
            "English": 80,
        },
        created_at=datetime.now(),
    )

    assert student.get_score("Math") == 90
    assert student.get_score("English") == 80

def test_get_score_not_found():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 90,
        },
        created_at=datetime.now(),
    )

    with pytest.raises(KeyError, match="Course not found"):
        student.get_score("English")

def test_average_score():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 90,
            "English": 80,
            "Python": 70,
        },
        created_at=datetime.now(),
    )

    assert student.average_score() == 80

def test_average_score_when_no_scores():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={},
        created_at=datetime.now(),
    )

    assert student.average_score() == 0

def test_is_passed():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 60,
            "English": 60,
        },
        created_at=datetime.now(),
    )

    assert student.is_passed() is True

def test_is_not_passed():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 59,
            "English": 59,
        },
        created_at=datetime.now(),
    )

    assert student.is_passed() is False


def test_to_dict():
    created_at = datetime(2026, 9, 30, 10, 0, 0)

    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 90,
            "English": 80,
        },
        created_at=created_at,
    )

    result = student.to_dict()

    print(result)

    assert result == {
        "name": "Tom",
        "student_id": 10001,
        "scores": {
            "Math": 90,
            "English": 80,
        },
        "created_at": created_at,
    }

def test_str():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 90
        },
        created_at=datetime(2026, 9, 30, 10, 0, 0),
    )

    result = str(student)

    print(f"Student name: {student}")

    assert "Tom" in result
    assert "10001" in result
    assert "Math" in result
    assert "90" in result

def test_repr():
    student = Student(
        name="Tom",
        student_id=10001,
        scores={
            "Math": 90
        },
        created_at=datetime(2026, 9, 30, 10, 0, 0),
    )

    assert repr(student) == str(student)

def test_student_count():

    assert Student.student_count == 0

    s1 = Student(name="Tom", student_id=10001, scores={}, created_at=datetime.now())
    s2 = Student(name="Tom", student_id=10001, scores={}, created_at=datetime.now())
    s3 = Student(name="Tom", student_id=10001, scores={}, created_at=datetime.now())

    assert Student.student_count == 3

def test_instance_cover_class():
    # 演示实例属性遮蔽类属性的情况
    stu = Student(name="Tom", student_id=10001, scores={}, created_at=datetime.now())

    print(stu.school_name)

    stu.school_name = "Tom school"
    print(Student.school_name)
    print(stu.school_name)
    assert stu.school_name == "Tom school"


    pass


# 测试类属性
def test_school_name():

    print()
    print(Student.school_name)
    Student.school_name  = "test school_name"
    print(Student.school_name)
    assert Student.school_name == "test school_name"

    pass


def test_from_dict():
    data =  {'name': 'Tom', 'student_id': 10001, 'scores': {'Math': 90, 'English': 80}}
    stu = Student.from_dict(data)
    assert stu.name == "Tom"
    assert stu.student_id == 10001
    assert stu.scores == {'Math': 90, 'English': 80}
    pass
