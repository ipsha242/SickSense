from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
import datetime

from sklearn.ensemble import GradientBoostingClassifier

from .forms import DoctorForm
from .models import *
from django.contrib.auth import authenticate, login, logout
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from django.http import HttpResponse, HttpResponseRedirect
# Create your views here.



from pickle import encode_long
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def Home(request):
    return render(request,'carousel.html')

def Admin_Home(request):
    dis = Search_Data.objects.all()
    pat = Patient.objects.all()
    doc = Doctor.objects.all()
    feed = Feedback.objects.all()

    d = {'dis':dis.count(),'pat':pat.count(),'doc':doc.count(),'feed':feed.count()}
    return render(request,'admin_home.html',d)

@login_required(login_url="login")
def assign_status(request,pid):
    doctor = Doctor.objects.get(id=pid)
    if doctor.status == 1:
        doctor.status = 2
        messages.success(request, 'Selected doctor are successfully withdraw his approval.')
    else:
        doctor.status = 1
        messages.success(request, 'Selected doctor are successfully approved.')
    doctor.save()
    return redirect('view_doctor')

@login_required(login_url="login")
def User_Home(request):
    return render(request,'patient_home.html')

@login_required(login_url="login")
def Doctor_Home(request):
    return render(request,'doctor_home.html')


def Login_User(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        sign = ""
        if user:
            try:
                sign = Patient.objects.get(user=user)
            except:
                pass
            if sign:
                login(request, user)
                error = "pat1"
            else:
                pure=False
                try:
                    pure = Doctor.objects.get(status=1,user=user)
                except:
                    pass
                if pure:
                    login(request, user)
                    error = "pat2"
                else:
                    login(request, user)
                    error="notmember"
        else:
            error="not"
    d = {'error': error}
    return render(request, 'login.html', d)

def Login_admin(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        if user.is_staff:
            login(request, user)
            error="pat"
        else:
            error="not"
    d = {'error': error}
    return render(request, 'admin_login.html', d)

def Signup_User(request):
    error = ""
    if request.method == 'POST':
        full = request.POST['fname']
        u = request.POST['uname']
        e = request.POST['email']
        p = request.POST['pwd']
        
        user = User.objects.create_user(email=e, username=u, password=p)
      
        Patient.objects.create(user=user, f_name=full)
       
        error = "create"
    d = {'error':error}
    return render(request,'register.html',d)

def Logout(request):
    logout(request)
    return redirect('home')

@login_required(login_url="login")
def Change_Password(request):
    sign = 0
    user = User.objects.get(username=request.user.username)
    error = ""
    if not request.user.is_staff:
        try:
            sign = Patient.objects.get(user=user)
            if sign:
                error = "pat"
        except:
            sign = Doctor.objects.get(user=user)
    terror = ""
    if request.method=="POST":
        n = request.POST['pwd1']
        c = request.POST['pwd2']
        o = request.POST['pwd3']
        if c == n:
            u = User.objects.get(username__exact=request.user.username)
            u.set_password(n)
            u.save()
            terror = "yes"
        else:
            terror = "not"
    d = {'error':error,'terror':terror,'data':sign}
    return render(request,'change_password.html',d)






@login_required(login_url="login")
def add_doctor(request,pid=None):
    doctor = None
    if pid:
        doctor = Doctor.objects.get(id=pid)
    if request.method == "POST":
        form = DoctorForm(request.POST, request.FILES, instance = doctor)
        if form.is_valid():
            new_doc = form.save()
            new_doc.status = 1
            if not pid:
                user = User.objects.create_user(password=request.POST['password'], username=request.POST['username'], first_name=request.POST['first_name'], last_name=request.POST['last_name'])
                new_doc.user = user
            new_doc.save()
            return redirect('view_doctor')
    d = {"doctor": doctor}
    return render(request, 'add_doctor.html', d)



@login_required(login_url="login")


@login_required(login_url="login")
def view_search_pat(request):
    doc = None
    try:
        doc = Doctor.objects.get(user=request.user)
        data = Search_Data.objects.filter(patient_address_icontains=doc.address).order_by('-id')
    except:
        try:
            doc = Patient.objects.get(user=request.user)
            data = Search_Data.objects.filter(patient=doc).order_by('-id')
        except:
            data = Search_Data.objects.all().order_by('-id')
    return render(request,'view_search_pat.html',{'data':data})

@login_required(login_url="login")
def delete_doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    doc.delete()
    return redirect('view_doctor')

@login_required(login_url="login")
def delete_feedback(request,pid):
    doc = Feedback.objects.get(id=pid)
    doc.delete()
    return redirect('view_feedback')

@login_required(login_url="login")
def delete_patient(request,pid):
    doc = Patient.objects.get(id=pid)
    doc.delete()
    return redirect('view_patient')

@login_required(login_url="login")
def delete_searched(request,pid):
    doc = Search_Data.objects.get(id=pid)
    doc.delete()
    return redirect('view_search_pat')

@login_required(login_url="login")
def View_Doctor(request):
    doc = Doctor.objects.all()
    d = {'doc':doc}
    return render(request,'view_doctor.html',d)

@login_required(login_url="login")
def View_Patient(request):
    patient = Patient.objects.all()
    d = {'patient':patient}
    return render(request,'view_patient.html',d)

@login_required(login_url="login")
def View_Feedback(request):
    dis = Feedback.objects.all()
    d = {'dis':dis}
    return render(request,'view_feedback.html',d)

@login_required(login_url="login")
def View_My_Detail(request):
    terror = ""
    user = User.objects.get(id=request.user.id)
    error = ""
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    d = {'error': error,'pro':sign}
    return render(request,'profile_doctor.html',d)

@login_required(login_url="login")
def Edit_Doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    error = ""
    # type = Type.objects.all()
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        cat = request.POST['type']
        try:
            im = request.FILES['image']
            doc.image=im
            doc.save()
        except:
            pass
        dat = datetime.date.today()
        doc.user.first_name = f
        doc.user.last_name = l
        doc.user.email = e
        doc.contact = con
        doc.category = cat
        doc.address = add
        doc.user.save()
        doc.save()
        error = "create"
    d = {'error':error,'doc':doc,'type':type}
    return render(request,'edit_doctor.html',d)

@login_required(login_url="login")
def Edit_My_deatail(request):
    terror = ""
    print("Hii welvome")
    user = User.objects.get(id=request.user.id)
    error = ""
    # type = Type.objects.all()
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        try:
            im = request.FILES['image']
            sign.image = im
            sign.save()
        except:
            pass
        to1 = datetime.date.today()
        sign.user.first_name = f
        sign.user.last_name = l
        sign.user.email = e
        sign.contact = con
        if error != "pat":
            cat = request.POST['type']
            sign.category = cat
            sign.save()
        sign.address = add
        sign.user.save()
        sign.save()
        terror = "create"
    d = {'error':error,'terror':terror,'doc':sign}
    return render(request,'edit_profile.html',d)

@login_required(login_url='login')
def sent_feedback(request):
    terror = None
    if request.method == "POST":
        username = request.POST['uname']
        message = request.POST['msg']
        username = User.objects.get(username=username)
        Feedback.objects.create(user=username, messages=message)
        terror = "create"
    return render(request, 'sent_feedback.html',{'terror':terror})

def add_coviddetail(request):
    input_data = []
    covidpred=0
    pred=None
      
    if request.method=="POST":
        
        DATA_PATH = Admin_Helath_CSV.objects.get(id=3)
        data = pd.read_csv(DATA_PATH.csv_file).dropna(axis = 1)
    
        
        X = data.columns
        y = data['covid_res']
        
        X_train, X_test, y_train, y_test =train_test_split(
        X, y, test_size = 0.2, random_state = 24)
        
        
        final_svm_model = SVC()
        final_nb_model = GaussianNB()
        final_rf_model = RandomForestClassifier(random_state=18)
        final_svm_model.fit(X, y)
        final_nb_model.fit(X, y)
        final_rf_model.fit(X, y)
        svm_preds = final_svm_model.predict(X_test)
        nb_preds = final_nb_model.predict(X_test)
        rf_preds = final_rf_model.predict(X_test)
        
        def predictCovid(symptoms):
            # print("All Symptoms = ", symptoms)
            # symptoms = symptoms.split(",")
            
            # # creating input data for the models
            input_data = []
            for symptom in symptoms:
               if symptom in X.columns:
                   input_data.append(1)
               else:
                   input_data.append(0)
                
            # generating individual outputs
            rf_prediction = final_rf_model.predict(input_data)[0]
            nb_prediction = final_nb_model.predict(input_data)[0]
            svm_prediction = final_svm_model.predict(input_data)[0]
            all_preds = [svm_preds, nb_preds, rf_preds]
            count_yes,  count_no,result=0
            for i in all_preds:
               if i == 1:
                  count_yes=count_yes+1
               else:
                  count_no=count_no+1
            
            if count_yes>count_no:
                pred="<span style='color:red'>You are likely to be Covid Positive. You should get checked.</span>"
            else:
                pred="<span style='color:green'>You are likely to be Covid Negative. No need to get checked.</span>"
        
    return render(request, 'add_coviddetail.html', {'pred':pred})
            
    
    



def add_genralhealth(request):
    predictiondata = None
    deseaseli = []
    if request.method=="POST":
        for i,j in request.POST.items():
            if "csrfmiddlewaretoken" != i:
                deseaseli.append(i)
        # training.csv
        DATA_PATH = Admin_Helath_CSV.objects.get(id=2)
        #df = pd.read_csv(csv_file.csv_file)
        #DATA_PATH = "c:/Users/bhuwa/OneDrive/Desktop/dataset/Training.csv"
        data = pd.read_csv(DATA_PATH.csv_file).dropna(axis = 1)

        # Checking whether the dataset is balanced or not
        disease_counts = data["prognosis"].value_counts()
        temp_df = pd.DataFrame({
            "Disease": disease_counts.index,
            "Counts": disease_counts.values
        })

        plt.figure(figsize = (18,8))
        sns.barplot(x = "Disease", y = "Counts", data = temp_df)
        plt.xticks(rotation=90)
        # plt.show()

        # Encoding the target value into numerical
        # value using LabelEncoder
        encoder = LabelEncoder()
        data["prognosis"] = encoder.fit_transform(data["prognosis"])


        X = data.iloc[:,:-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test =train_test_split(
        X, y, test_size = 0.2, random_state = 24)



        symptoms = X.columns.values
        symptom_index = {}
        for index, value in enumerate(symptoms):
            symptom = " ".join([i.capitalize() for i in value.split("_")])
            symptom_index[symptom] = index
        
        data_dict = {
            "symptom_index":symptom_index,
            "predictions_classes":encoder.classes_
        }

        final_svm_model = SVC()
        final_nb_model = GaussianNB()
        final_rf_model = RandomForestClassifier(random_state=18)
        final_svm_model.fit(X, y)
        final_nb_model.fit(X, y)
        final_rf_model.fit(X, y)

        #Testing.csv
        DATA_PATH2 = Admin_Helath_CSV.objects.get(id=3)
        test_data = pd.read_csv(DATA_PATH2.csv_file).dropna(axis=1)

        test_X = test_data.iloc[:, :-1]
        test_Y = encoder.transform(test_data.iloc[:, -1])

        svm_preds = final_svm_model.predict(test_X)
        nb_preds = final_nb_model.predict(test_X)
        rf_preds = final_rf_model.predict(test_X)

        final_preds = [mode([i,j,k])[0][0] for i,j,
                    k in zip(svm_preds, nb_preds, rf_preds)]

        print(f"Accuracy on Test dataset by the combined model\
        : {accuracy_score(test_Y, final_preds)*100}")

        cf_matrix = confusion_matrix(test_Y, final_preds)
        plt.figure(figsize=(12,8))

        sns.heatmap(cf_matrix, annot = True)
        # plt.title("Confusion Matrix for Combined Model on Test Dataset")
        # # plt.show()

        def predictDisease(symptoms):
            # print("All Symptoms = ", symptoms)
            # symptoms = symptoms.split(",")
            
            # # creating input data for the models
            input_data = [0] * len(data_dict["symptom_index"])
            for symptom in symptoms:
                index = data_dict["symptom_index"][symptom]
                input_data[index] = 1
                
            # reshaping the input data and converting it
            # into suitable format for model predictions
            input_data = np.array(input_data).reshape(1,-1)
            
            # generating individual outputs
            rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
            nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
            svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
            
            # making final prediction by taking mode of all predictions
            final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
            predictions = {
                "Final Prediction":final_prediction
            }
            return predictions
        '''disease_to_doctor={
'Fungal infection':['General Physician','Dermatologist'],
'Allergy':['General Physician','Dermatologist','Pulmonologist'],
'GERD':['General Physician','Gastroenterologist'],
'Chronic cholestasis':['General Physician','Hepatologist'],
'Drug Reaction':['General Physician','Dermatologist'],
'Peptic ulcer diseae':['General Physician','Gastroenterologist'],
'AIDS':['General Physician','Infectious Disease Specialist'],
'Diabetes':['General Physician','Endocrinologist'],
'Gastroenteritis':['General Physician','Gastroenterologist'],
'Bronchial Asthma':['General Physician','Pulmonoloist'],
'Hypertension':['General Physician','Cardiologist'],
'Migraine':['General Physician','Neurologist'],
'Cervical spondylosis':['General Physician','Orthopedic'],
'Paralysis (brain hemorrhage)':['General Physician','Neurologist'],
'Jaundice':['General Physician','Hepatologist','Gastroenterologist'],
'Malaria':['General Physician','Infectious Disease Specialist'],
'Chicken pox':['General Physician','Dermatologist'],
'Dengue':['General Physician','Infectious Disease Specialist'],
'Typhoid':['General Physician','Gastroenterologist'],
'hepatitis A':['General Physician','Hepatologist'],
'Hepatitis B':['General Physician','Hepatologist'],
'Hepatitis C':['General Physician','Hepatologist'],
'Hepatitis D':['General Physician','Hepatologist'],
'Hepatitis E':['General Physician','Hepatologist'],
'Alcoholic hepatitis':['General Physician','Hepatologist'],
'Tuberculosis':['General Physician','Pulmonoloist'],
'Common Cold':['General Physician','Pulmonoloist'],
'Pneumonia':['General Physician','Pulmonoloist'],
'Dimorphic hemmorhoids(piles)':['General Physician','Gastroenterologist'],
'Heartattack':['General Physician','Cardiologist'],
'Varicoseveins':['General Physician','Dermatologist'],
'Hypothyroidism':['General Physician','Endocrinologist'],
'Hyperthyroidism':['General Physician','Endocrinologist'],
'Hypoglycemia':['General Physician','Endocrinologist'],
'Osteoarthristis':['General Physician','Orthopedic'],
'Arthritis':['General Physician','Orthopedic'],
'(vertigo) Paroymsal  Positional Vertigo':['General Physician','Neurologist'],
'Acne':['General Physician','Dermatologist'],
'Urinary tract infection':['General Physician','Urologist'],
'Psoriasis':['General Physician','Dermatologist'],'Impetigo':['General Physician','Dermatologist']}
        
        def find_relevant_doctors(disease):
            relevant_doctors = disease_to_doctor.get(disease, [])
            return relevant_doctors'''

        # Testing the function
        predictiondata = predictDisease(deseaseli)
        '''doctors=find_relevant_doctors(predictiondata)'''
        patient = Patient.objects.get(user=request.user)
        Search_Data.objects.create(patient=patient, prediction_accuracy=round(accuracy_score(test_Y, final_preds)*100,2), result=predictiondata["Final Prediction"], values_list=deseaseli, predict_for="General Health Prediction")
        
        # print(deseaseli)
    alldisease = ['Itching','Skin Rash','Nodal Skin Eruptions','Continuous Sneezing','Shivering','Chills','Joint Pain',	'Stomach Pain','Acidity','Ulcers On Tongue','Muscle Wasting','Vomiting','Burning Micturition','Spotting Urination','Fatigue','Weight Gain','Anxiety','Cold Hands And Feets','Mood Swings','Weight Loss','Restlessness','Lethargy','Patches In Throat','Irregular Sugar Level','Cough','High Fever','Sunken Eyes','Breathlessness','Sweating','Dehydration',	'Indigestion','Headache','Yellowish Skin','Dark Urine','Nausea','Loss Of Appetite','Pain Behind The Eyes','Back Pain','Constipation','Abdominal Pain','Diarrhoea','Mild Fever','Yellow Urine','Yellowing Of Eyes','Acute Liver Failure','Fluid Overload','Swelling Of Stomach','Swelled Lymph Nodes','Malaise','Blurred And Distorted Vision','Phlegm','Throat Irritation','Redness Of Eyes','Sinus Pressure','Runny Nose','Congestion','Chest Pain','Weakness In Limbs','Fast Heart Rate',	'Pain During Bowel Movements','Pain In Anal Region','Bloody Stool','Irritation In Anus','Neck Pain','Dizziness','Cramps','Bruising','Obesity','Swollen Legs','Swollen Blood Vessels','Puffy Face And Eyes','Enlarged Thyroid','Brittle Nails','Swollen Extremeties','Excessive Hunger','Extra Marital Contacts','Drying And Tingling Lips','Slurred Speech','Knee Pain','Hip Joint Pain','Muscle Weakness','Stiff Neck','Swelling Joints','Movement Stiffness','Spinning Movements','Loss Of Balance','Unsteadiness','Weakness Of One Body Side','Loss Of Smell','Bladder Discomfort','Continuous Feel Of Urine','Passage Of Gases','Internal Itching','Toxic Look (Typhos)',	'Depression','Irritability','Muscle Pain','Altered Sensorium','Red Spots Over Body','Belly Pain','Abnormal Menstruation','Dischromic Patches','Watering From Eyes','Increased Appetite','Polyuria','Family History','Mucoid Sputum','Rusty Sputum','Lack Of Concentration',	'Visual Disturbances','Receiving Blood Transfusion','Receiving Unsterile Injections','Coma','Stomach Bleeding',	'Distention Of Abdomen','History Of Alcohol Consumption','Fluid Overload','Blood In Sputum','Prominent Veins On Calf','Palpitations','Painful Walking','Pus Filled Pimples', 'Blackheads','Scurring','Skin Peeling','Silver Like Dusting','Small Dents In Nails','Inflammatory Nails','Blister','Red Sore Around Nose','Yellow Crust Ooze']
    return render(request,'add_genralhealth.html', {'alldisease':alldisease, 'predictiondata':predictiondata})



def search_blood(request):
    data = Blood_Donation.objects.filter(status="Approved")
    if request.method == "POST":
        bg = request.POST['bg']
        place = request.POST['place']
        user = Patient.objects.get(user=request.user)
        Blood_Donation.objects.create(blood_group=bg, user=user, purpose="Request for Blood", status="Pending", place=place)
        messages.success(request, "Request Generated.")
        return redirect('search_blood')
    return render(request, 'search_blood.html', {'data':data})


def donate_blood(request):
    if request.method == "POST":
        bg = request.POST['bg']
        place = request.POST['place']
        user = Patient.objects.get(user=request.user)
        data = Blood_Donation.objects.create(blood_group=bg, user=user, purpose="Blood Donor", status="Pending", place=place)
        messages.success(request, "Added Your Detail.")
        return redirect('donate_blood')
    return render(request, 'donate_blood.html')

def request_blood(request):
    mydata = request.GET.get('action',0)
    data = Blood_Donation.objects.filter(purpose="Request for Blood")
    if mydata:
        data = data.filter(status=mydata)
    return render(request, 'request_blood.html', {'data':data})


@login_required(login_url="login")
def donator_blood(request):
    mydata = request.GET.get('action',0)
    data = Blood_Donation.objects.filter(purpose="Blood Donor")
    if mydata:
        data = data.filter(status=mydata)
    return render(request, 'donator_blood.html', {'data':data})

def change_status(request,pid):
    data = Blood_Donation.objects.get(id=pid)
    url = request.GET.get('data')
    if data.status == "Approved":
        data.status = "Pending"
        data.save()
    else:
        data.status = "Approved"
        data.save()
    return HttpResponseRedirect(url)

def appointment(request):
    if request.method == 'POST':
        full = request.POST['full']
        gender = request.POST['gender']
        age = request.POST['age']
        num = request.POST['num']
        department = request.POST['department']
        con_date_time = request.POST['con_date_time']
        
        Appointment.objects.create(fullname = full, gender = gender, age = age, num = num, department = department, con_date_time = con_date_time)
        terror = "create"
        return redirect('consultation_hist')
    return render(request, 'appointment.html')

def appointment_status(request, pid):
    data = Appointment.objects.get(id-pid)
    url = request.GET.get('data')
    if data.status == "Booking":
        data.status = "Pending"
        data.save()
    else:
        data.status = "Booked"
        data.save()
    return HttpResponseRedirect(url)

@login_required(login_url="login")
def View_Doctor_Patient(request):
    doc = Doctor.objects.all()
    d = {'doc':doc}
    return render(request,'view_pat_doctors.html',d)

def Remedies(request):
       return render(request,'remedies.html')
   
   
