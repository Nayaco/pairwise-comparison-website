from django.db import models
from django.utils import timezone
import datetime
# Create your models here.

from django.db import models




class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')
    def __str__(self):
        return self.question_text
    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)



class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)
    def __str__(self):
        return self.choice_text

class User(models.Model):
    ip = models.CharField(max_length = 20) # ip20位够了吧
    num_compare_cnt = models.IntegerField(default=0) # > 40则不能继续填
    age = models.IntegerField(default=0) # 年龄段
    familiarity = models.IntegerField(default=0) # 对上海的熟悉程度

    def __str__(self):
        return self.ip

class Preference_livability(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    A = models.IntegerField(default=-1) # 年龄段
    B = models.IntegerField(default=-1) # 年龄段
    pref = models.BooleanField(default=False) # 年龄段
    time = models.DateTimeField('date annotated', default=timezone.now())
    def __str__(self):
        return str(self.A) + ', ' + str(self.B)


class Preference_resource(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    A = models.IntegerField(default=-1) # 年龄段
    B = models.IntegerField(default=-1) # 年龄段
    pref = models.BooleanField(default=False) # 年龄段
    time = models.DateTimeField('date annotated', default=timezone.now())
    def __str__(self):
        return str(self.A) + ', ' + str(self.B)

class Latest_comparison(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    A = models.IntegerField(default=-1) # 年龄段
    B = models.IntegerField(default=-1) # 年龄段