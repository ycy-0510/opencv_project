from django.shortcuts import render


def index(request):
    """Return the static default dashboard page (no dynamic data)."""
    return render(request, 'dashboard/index.html')
