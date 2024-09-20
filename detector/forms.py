from django import forms
from .models import SignImage


class SignImageForm(forms.ModelForm):
    class Meta:
        model = SignImage
        fields = ['image']
