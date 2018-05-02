from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import FileForm

from .algorithms import NewsChecker

def index(request):
    if request.method == 'POST':
        form = FileForm(request.POST)
        if form.is_valid():
            file_text = form.cleaned_data.get('file_text', None)
            algorithm = form.cleaned_data.get('algorithm', None)

            predicted = NewsChecker.fake_or_real(file_text, int(algorithm))
            is_real = (predicted[0] == 'REAL')
            return render(request, 'index.html', {'form': form, 'is_real': is_real})

    else:
        form = FileForm()
    return render(request, 'index.html', {'form': form})
