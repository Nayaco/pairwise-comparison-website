from django.contrib import admin
from polls.models import Question,Choice,User,Preference_livability, Preference_resource, Latest_comparison

class ChoiceInline(admin.StackedInline):
    model = Choice
    extra = 3

class QuestionAdmin(admin.ModelAdmin):
    fieldsets = [
        (None,    {'fields': ['question_text']}),
        ('Date information', {'fields': ['pub_date'],'classes':['collapse']}),
    ]
    inlines = [ChoiceInline]
    
    list_display = ('question_text','pub_date','was_published_recently')
    search_fields = ['question_text']

class Preference_livability_Inline(admin.StackedInline):
    model = Preference_livability   
    extra = 0
class Preference_resource_Inline(admin.StackedInline):
    model = Preference_resource    
    extra = 0
class Latest_comparison_Inline(admin.StackedInline):
    model = Latest_comparison    
    extra = 0
class UserAdmin(admin.ModelAdmin):
    fieldsets = [
        (None,    {'fields': ['ip']}),
    ]
    inlines = [Preference_livability_Inline, Preference_resource_Inline, Latest_comparison_Inline]   
    list_display = ('ip','num_compare_cnt','age', 'familiarity')
    search_fields = ['ip']

admin.site.register(Question,QuestionAdmin)
admin.site.register(User,UserAdmin)
admin.site.register(Choice)

