# Generated by Django 3.1.3 on 2023-05-21 18:03

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('healthapp', '0005_auto_20230521_2009'),
    ]

    operations = [
        migrations.RenameField(
            model_name='doctor',
            old_name='full_name',
            new_name='email',
        ),
        migrations.RenameField(
            model_name='doctor',
            old_name='reg_id',
            new_name='experience',
        ),
        migrations.AddField(
            model_name='doctor',
            name='user',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]
