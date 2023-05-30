# Generated by Django 3.1.3 on 2023-05-21 14:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('healthapp', '0004_appointment'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='doctor',
            name='dob',
        ),
        migrations.RemoveField(
            model_name='doctor',
            name='doj',
        ),
        migrations.RemoveField(
            model_name='doctor',
            name='user',
        ),
        migrations.AddField(
            model_name='doctor',
            name='fees',
            field=models.CharField(max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='doctor',
            name='full_name',
            field=models.CharField(max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='doctor',
            name='reg_id',
            field=models.CharField(max_length=100, null=True),
        ),
    ]
