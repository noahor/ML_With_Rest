from aiohttp import web 
import json
import asyncio 
import async_timeout
from ML import *
from io import StringIO
' create a full REST api server via Python ' 


async def handle(request):
    response_obj = {'status': 'success','message':'WellCome to Zhack Macheing learning'}
    return web.Response(text= json.dumps(response_obj),status=200)


# Predict
async def predictJob(request):
    try:
        jsonResult = await request.json()

        result = predict()
        response_obj = {'status': 'predict Job success','score':str(result)}
        headers = {"Content-Type" : "application/json"}
        return web.Response (text =json.dumps(response_obj),status=200, headers=headers)

    except Exception as e:
        response_obj = {'status': 'predict Job failed','message' : str(e)}
        return web.Response (text =json.dumps(response_obj),status=500)

# Predict
async def saveModel(request):
    try:
        jsonResult = await request.json()
        path = jsonResult["path"]
        save(path)
        response_obj = {'status': 'Model has been save  success','path':str(path)}
        headers = {"Content-Type" : "application/json"}
        return web.Response (text =json.dumps(response_obj),status=200, headers=headers)

    except Exception as e:
        response_obj = {'status': 'failed to save the model','message' : str(e)}
        return web.Response (text =json.dumps(response_obj),status=500)


# Predict
async def LoadModel(request):
    try:
        jsonResult = await request.json()
        path = jsonResult["path"]
        load_model_from_path(path)
       
       
        response_obj = {'status': 'predict Job success','score':6,'IsNormal':6}
        headers = {"Content-Type" : "application/json"}
        return web.Response (text =json.dumps(response_obj),status=200, headers=headers)

    except Exception as e:
        response_obj = {'status': 'predict Job failed','message' : str(e)}
        return web.Response (text =json.dumps(response_obj),status=500)


# Train
async def TrainModel(request):
    try:
        jsonResult = await request.json()
        Train_results = Train() 
        
        response_obj = {'status': 'TrainModel succesed', "Train model evaluation" : Train_results['test_error_rate'],"Saved Model path is" : Train_results['path']}

        headers = {"Content-Type" : "application/json"}
        return web.Response (text =json.dumps(response_obj),status=200, headers=headers)

    except Exception as e:
        response_obj = {'status': 'TrainModel failed','message' : str(e)}
        return web.Response (text =json.dumps(response_obj),status=500)

if __name__ == '__main__':
   app = web.Application()
   app.router.add_get('/',handle)
   app.router.add_post('/predictJob',predictJob)
   app.router.add_post('/TrainModel',TrainModel)
   app.router.add_post('/LoadModel',LoadModel)
   app.router.add_post('/saveModel',saveModel)
   
   
   print("Running app")
 
   web.run_app(app,port=8077)
  