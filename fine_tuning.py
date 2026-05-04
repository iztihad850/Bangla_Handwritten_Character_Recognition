def fine_tune(model, model_name, state="none"):

    

    if(state == "partial"):
        if(model_name == "mobilenet_v2"):
            for param in model.features[-3:].parameters():
                param.requires_grad = True
        elif(model_name == "resnet_50"):
            for param in model.layer4.parameters():
                param.requires_grad = True
        elif(model_name == "efficientnet_b2" or model_name == "efficientnet_b4"):
            for param in model.features[-2:].parameters():
                param.requires_grad = True
        elif(model_name == "tiny_vit" or model_name == "mobilevit_s" or model_name == "efficientvit"):
            for param in model.stages[-1].parameters():
                param.requires_grad = True

    elif(state == "full"):
        for params in model.parameters():
            params.requires_grad = True

    return model