import json
import gc
import numpy as np

from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from django.utils import timezone


from .test_server import get_pair, update_params, RecalcParams, get_mu

from .models import Choice, Question, User, Preference_livability, Preference_resource, Latest_comparison

lon1 = 120.85
lon2 = 121.98
lat1 = 30.68
lat2 = 31.88

n_lat = 133
n_lon = 107

step_lat = (lat2 - lat1) / n_lat
step_lon = (lon2 - lon1) / n_lon

def get_age_id(age):
    return {
        '18岁以下':0,
        '18-25岁':1,
        '26-30岁':2,
        '31-35岁':3,
        '36-40岁':4,
        '41-45岁':5,
        '46-50岁':6,
        '51-55岁':7,
        '56-60岁':8,
        '61-65岁':9,
        '65岁以上':10,
    }.get(age,'error') 

def get_familarity_id(familarity):
    return {
        '非常熟悉（e.g. 久居上海）':0,
        '比较熟悉（e.g. 在上海求学、短暂工作）':1,
        '不太熟悉（e.g. 旅游来过上海）':2,
        '对上海完全不了解':3,
    }.get(familarity,'error') 

class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        """Return the last five published questions."""
        # 在这里先get一下ip，然大家填写几个小问题：1. 年龄段， 2. 对上海的熟悉程度
        return Question.objects.order_by('-pub_date')[:5] 


class DetailView(generic.DetailView):
    model = Question
    template_name = 'polls/detail.html'

def detail_getpair(request, pk):
    question_id = pk
    if question_id == 1 or question_id == 2:
        pass
    else:
        question = get_object_or_404(Question, pk=question_id)
        # 获取IP
        if request.META.__contains__('HTTP_X_FORWARDED_FOR'):
            ip =  request.META['HTTP_X_FORWARDED_FOR']
        elif request.META.__contains__('REMOTE_ADDR'):
            ip = request.META['REMOTE_ADDR']
        else:
            ip = 'unknown'

        # 如果是新IP则加入数据库，如果老ip答题超过50次则停止
        this_user = User.objects.filter(ip=ip)
        if len(this_user) == 0: # new ip
            User.objects.create(ip=ip)
            this_user = User.objects.filter(ip=ip)
        else:
            if this_user[0].num_compare_cnt > 50:
                return render(request, 'polls/detail.html', {# 你不能再做题了！
                    'question': question,
                    'error_message': "本IP答题次数超过50次，无法继续答题。感谢您的参与！",
                    })

        # 答题超过10次可以提醒一下，说well done! brother
        if question_id == 3: # resource
            selected_pair = get_pair() # 这里get_pair分两种
            print("output selected pair", selected_pair)
            # 这里处理一下！！！ 把selected pair变成经纬度
            out_list_500 = np.load('polls/out_list_1000.npy')

            A = out_list_500[selected_pair[0]]
            B = out_list_500[selected_pair[1]]

            location_pair = [ [A[0] * step_lat + lat1, A[1] * step_lon + lon1], 
                            [B[0] * step_lat + lat1, B[1] * step_lon + lon1]]

            if len(Latest_comparison.objects.filter(user=User.objects.filter(ip=ip)[0])) == 0:

                Latest_comparison.objects.create(\
                    user=User.objects.filter(ip=ip)[0], \
                    A=selected_pair[0], \
                    B=selected_pair[1])
            else:
                Latest_comparison.objects.filter(user=User.objects.filter(ip=ip)[0]).update(A=selected_pair[0], B=selected_pair[1])
            
            
            del out_list_500
            gc.collect()

            if this_user[0].num_compare_cnt >= 10 and ip != 'unknown':
            
                return render(request, 'polls/detail.html', {# 你不能再做题了！
                        'question': question,
                        'compare_list': location_pair,
                        # 'error_message': "您已经进行了{}次比较".format(res[0].num_compare_cnt),
                        'error_message': "本IP已经进行了{}次回答（大于10次），谢谢您的配合！支付宝口令红包：xxxxx ，先到先得。您也可以继续答题".format(this_user[0].num_compare_cnt),
                        'keep_going': "您也可以点此继续答题，感谢您的参与！",
                        })
            elif ip != 'unknown':
                return render(request, 'polls/detail.html', {# 你不能再做题了！
                        'question': question,
                        'compare_list': location_pair,
                        'error_message': "您已经进行了{}次比较".format(this_user[0].num_compare_cnt),
                        # 'keep_going': "您也可以点此继续答题，感谢您的参与！",
                        })
            else:
                return render(request, 'polls/detail.html', {# 你不能再做题了！
                        'question': question,
                        'compare_list': location_pair,
                        # 'keep_going': "您也可以点此继续答题，感谢您的参与！",
                        })





class ResultsView(generic.DetailView):
    model = Question
    template_name = 'polls/results.html'

# from __future__ import unicode_literals


def initvote(request):
    try:
        selected_choice_age = question.choice_set.get(pk=request.POST['choice0'])
        selected_choice_fami = question.choice_set.get(pk=request.POST['choice1'])
    except:
        # 没填也没关系
        return HttpResponseRedirect(reverse('polls:detail', args=(3,))) 
    else:
        if request.META.__contains__('HTTP_X_FORWARDED_FOR'):
            ip =  request.META['HTTP_X_FORWARDED_FOR']
        elif request.META.__contains__('REMOTE_ADDR'):
            ip = request.META['REMOTE_ADDR']
        else:
            ip = 'unknown'

        # 如果是新IP则加入数据库，如果老ip答题超过50次则停止
        this_user = User.objects.filter(ip=ip)
        if len(this_user) == 0: # new ip
            User.objects.create(ip=ip)
            this_user = User.objects.filter(ip=ip)
            # 新ip来到时才增加
            selected_choice.votes += 1
            selected_choice.save()
        else:
            # print(this_user[0].num_compare_cnt)
            if this_user[0].num_compare_cnt > 50 and ip != 'unknown':
                return render(request, 'polls/detail.html', {# 你不能再做题了！
                    'question': question,
                    'error_message': "本IP答题次数超过50次，无法继续答题。感谢您的参与！",
                    })
        this_user.update(age=get_age_id(selected_choice_age.__str__()))
        this_user.update(familiarity=get_familarity_id(selected_choice_fami.__str__()))
        return HttpResponseRedirect(reverse('polls:detail', args=(3,))) 




def vote(request, question_id): 
    question = get_object_or_404(Question, pk=question_id)

    try:
        print(request.POST['choice'])
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
        print(selected_choice)
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "你没有选择任何选项",
        })
    else:
        # 获取IP
        if request.META.__contains__('HTTP_X_FORWARDED_FOR'):
            ip =  request.META['HTTP_X_FORWARDED_FOR']
        elif request.META.__contains__('REMOTE_ADDR'):
            ip = request.META['REMOTE_ADDR']
        else:
            ip = 'unknown'

        # 如果是新IP则加入数据库，如果老ip答题超过50次则停止
        this_user = User.objects.filter(ip=ip)
        if len(this_user) == 0: # new ip
            User.objects.create(ip=ip)
            this_user = User.objects.filter(ip=ip)
            # 新ip来到时才增加
            selected_choice.votes += 1
            selected_choice.save()
        else:
            print(this_user[0].num_compare_cnt)
            if this_user[0].num_compare_cnt > 50 and ip != 'unknown':
                return render(request, 'polls/detail.html', {# 你不能再做题了！
                    'question': question,
                    'error_message': "本IP答题次数超过50次，无法继续答题。感谢您的参与！",
                    })
            

        if question_id == 1: # 年龄信息，一上来首先填好的
            this_user.update(age=get_age_id(selected_choice.__str__()))
            return HttpResponseRedirect(reverse('polls:detail', args=(2,)))

        elif question_id == 2: # 是否在上海ok的信息
            this_user.update(familiarity=get_familarity_id(selected_choice.__str__()))
            return HttpResponseRedirect(reverse('polls:detail', args=(3,))) # 或者是1
        else: # 其他的
            if question_id == 3: # resource value
                A = Latest_comparison.objects.filter(user=this_user[0])[0].A
                B = Latest_comparison.objects.filter(user=this_user[0])[0].B
                pref = True if selected_choice.__str__()=='A' else False
                
                # 添加就完事了
                Pref = Preference_resource.objects.create(user=this_user[0], A=A, B=B, pref=pref, time=timezone.now())

                # update paramaters
                print([(A, B)], 'aaa')
                update_params([(A, B)], pref)
                this_user.update(num_compare_cnt=len(Preference_resource.objects.filter(user=this_user[0])))

                # in fact这里需要mutex一下
                
            # elif question_id == 1: # livability
            #     A = Latest_comparison.objects.filter(user=this_user[0])[0].A
            #     B = Latest_comparison.objects.filter(user=this_user[0])[0].B
            #     pref = True if selected_choice.__str__()=='A' else False
            #     Pref = Preference_livability.objects.create(user=this_user[0], A=A, B=B, pref=pref)

            #     # update paramaters
            #     print([(A, B)], 'aaa')
            #     update_params([(A, B)], pref)

            if this_user[0].num_compare_cnt == 10 and ip != 'unknown':
                # 可以显示红包了，然后也显示一下可以答另一个题

                return render(request, 'polls/detail.html', {# 你不能再做题了！
                    'question': question,
                    'error_message': "本IP已经进行了至少10次回答，谢谢您的配合！支付宝口令红包：xxxxx ，先到先得。您也可以继续答题",
                    'keep_going': "您也可以点此继续答题，感谢您的参与！",
                    })

            return HttpResponseRedirect(reverse('polls:detail', args=(question.id, ))) # 返回它自己
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.

# 还得写个程序把所有的pref全部读出来啊！！
def getpair(request, question_id):
    question = get_object_or_404(Question, pk=question_id)

    try:
        # selected_choice = question.choice_set.get(pk=request.POST['choice'])
        1
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        # 获取IP
        if request.META.__contains__('HTTP_X_FORWARDED_FOR'):
            ip =  request.META['HTTP_X_FORWARDED_FOR']
        elif request.META.__contains__('REMOTE_ADDR'):
            ip = request.META['REMOTE_ADDR']
        else:
            ip = 'unknown'
        # 如果是新IP则加入数据库，如果老ip答题超过50次则停止
        this_user = User.objects.filter(ip=ip)
        if len(this_user) == 0: # new ip
            User.objects.create(ip=ip)
        else:
            if this_user[0].num_compare_cnt > 50:
                return render(request, 'polls/detail.html', {# 你不能再做题了！
                    'question': question,
                    'error_message': "本IP答题次数超过50次，无法继续答题。感谢您的参与！",
                    })

        # 答题超过10次可以提醒一下，说well done! brother
        if question_id == 3: # resource
            selected_pair = get_pair() # 这里get_pair分两种
            print("output selected pair", selected_pair)
            # 这里处理一下！！！ 把selected pair变成经纬度
            out_list_500 = np.load('polls/out_list_1000.npy')

            A = out_list_500[selected_pair[0]]
            B = out_list_500[selected_pair[1]]

            location_pair = [ [A[0] * step_lat + lat1, A[1] * step_lon + lon1], 
                            [B[0] * step_lat + lat1, B[1] * step_lon + lon1]]

            if len(Latest_comparison.objects.filter(user=User.objects.filter(ip=ip)[0])) == 0:

                Latest_comparison.objects.create(\
                    user=User.objects.filter(ip=ip)[0], \
                    A=selected_pair[0], \
                    B=selected_pair[1])
            else:
                Latest_comparison.objects.filter(user=User.objects.filter(ip=ip)[0]).update(A=selected_pair[0], B=selected_pair[1])
            
            
            del out_list_500
            gc.collect()

            if this_user[0].num_compare_cnt >= 10 and ip != 'unknown':
            
                return render(request, 'polls/detail.html', {# 你不能再做题了！
                        'question': question,
                        'compare_list': location_pair,
                        # 'error_message': "您已经进行了{}次比较".format(res[0].num_compare_cnt),
                        'error_message': "本IP已经进行了{}次回答（大于10次），谢谢您的配合！支付宝口令红包：xxxxx ，先到先得。您也可以继续答题".format(this_user[0].num_compare_cnt),
                        'keep_going': "您也可以点此继续答题，感谢您的参与！",
                        })
            elif ip != 'unknown':
                return render(request, 'polls/detail.html', {# 你不能再做题了！
                        'question': question,
                        'compare_list': location_pair,
                        'error_message': "您已经进行了{}次比较".format(this_user[0].num_compare_cnt),
                        # 'keep_going': "您也可以点此继续答题，感谢您的参与！",
                        })
            else:
                return render(request, 'polls/detail.html', {# 你不能再做题了！
                        'question': question,
                        'compare_list': location_pair,
                        # 'keep_going': "您也可以点此继续答题，感谢您的参与！",
                        })
        # else: # livability
        #     selected_pair = get_pair() # 这里get_pair分两种
        #     print("output selected pair", selected_pair)
        #     if len(Latest_comparison.objects.filter(user=User.objects.filter(ip=ip)[0])) == 0:

        #         Latest_comparison.objects.create(\
        #             user=User.objects.filter(ip=ip)[0], \
        #             A=selected_pair[0], \
        #             B=selected_pair[1])
        #     else:
        #         Latest_comparison.objects.filter(user=User.objects.filter(ip=ip)[0]).update(A=selected_pair[0], B=selected_pair[1])

        #     return HttpResponseRedirect(reverse('polls:detail', args=(5,)))             
        
        # pass # TODO
        # return HttpResponseRedirect(reverse('polls:detail', args=(question.id,))) # 返回它自己

        # if question_id == 5: # resource value
        #     A = User.objects.filter(ip=ip)[0].Latest_comparison.A
        #     B = User.objects.filter(ip=ip)[0].Latest_comparison.B
        #     pref = True if selected_choice.__str__()=='A' else False
        #     Pref = Preference_resource.objects.create(user=User.objects.filter(ip=ip)[0], A=A, B=B, pref=pref)

        #     # update paramaters
        #     print([(A, B)], 'aaa')
        #     update_params([(A, B)], pref)

        #     # in fact这里需要mutex一下
            
        # elif question_id == 1: # livability
        #     A = Latest_comparison.objects.filter(user=User.objects.filter(ip=ip)[0])[0].A
        #     B = Latest_comparison.objects.filter(user=User.objects.filter(ip=ip)[0])[0].B
        #     pref = True if selected_choice.__str__()=='A' else False
        #     Pref = Preference_livability.objects.create(user=User.objects.filter(ip=ip)[0], A=A, B=B, pref=pref)

        #     # update paramaters
        #     print([(A, B)], 'aaa')
        #     update_params([(A, B)], pref)
            # update paramaters

            # in fact这里需要mutex一下

        # run recommendation codes

        # 把select出来的装进最新数据库里面去

def recalresult(request, question_id):
    # traverse database data
    question = get_object_or_404(Question, pk=question_id)

    try:
        # selected_choice = question.choice_set.get(pk=request.POST['choice'])
        1
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else: 
        allpref = Preference_resource.objects.order_by('time')

        compare_list = []
        pref_list    = []
        for item in allpref:
            compare_list.append([(item.A, item.B)])
            pref_list.append(item.pref)
        # 按时间顺序排个序
        RecalcParams(compare_list, pref_list) 
        
        mu = get_mu()

        return render(request, 'polls/results.html', {
                'question': question,
                'mumesg': str(mu),
                })
    # call a function to recalculate all data
    

def getparam(request, question_id):
    # load最新的params
    # 在这里也可以搞一个统计年龄的。。。

    question = get_object_or_404(Question, pk=question_id)
    
    agetext_list = [
        '18岁以下',
        '18-25岁',
        '26-30岁',
        '31-35岁',
        '36-40岁',
        '41-45岁',
        '46-50岁',
        '51-55岁',
        '56-60岁',
        '61-65岁',
        '65岁以上',
    ]
    age_list = []
    for i in range(len(agetext_list)):
        age_list.append(agetext_list[i] + " --- " + str(len(User.objects.filter(age=i))))
    
    familiar_list = [
            '非常熟悉（e.g. 久居上海）',
        '比较熟悉（e.g. 在上海求学、短暂工作）',
        '不太熟悉（e.g. 旅游来过上海）',
        '对上海完全不了解',
        ]
    fam_list = []
    for i in range(len(familiar_list)):
        fam_list.append(familiar_list[i] + " --- " + str(len(User.objects.filter(familiarity=i))))
    
    numprefs = len(Preference_resource.objects.filter())

    numips   = len(User.objects.filter())

    mu = get_mu()
    try:
        # selected_choice = question.choice_set.get(pk=request.POST['choice'])
        1
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        return render(request, 'polls/results.html', {
            'question': question,
            'mumesg': str(mu),
            'agemesg': age_list,
            'fammesg': fam_list,
            'numprefs': numprefs,
            'numips': numips,
            })