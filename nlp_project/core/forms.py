from django import forms
from django.core.validators import FileExtensionValidator

from .algorithms import NewsChecker


class FileForm(forms.Form):
    ALGORITHM_CHOICES = (
        (NewsChecker.ALGORITHM_MultinomialNB, 'Multinomial NB'),
        (NewsChecker.ALGORITHM_KNeighborsClassifier, 'KNeighbors Classifier'),
    )

    file_text = forms.CharField(widget=forms.Textarea(attrs={'width': "100%", 'cols': "90", 'rows': "12", }),
                                required=True)
    algorithm = forms.ChoiceField(widget=forms.RadioSelect, choices=ALGORITHM_CHOICES)

    # def clean(self):
    #     cleaned_data = super().clean()
    #     file = cleaned_data.get('file', None)
    #     file_text = cleaned_data.get('file_text', None)
    #     if not file and not file_text:
    #         raise forms.ValidationError("Either a file or a text should be entered")
