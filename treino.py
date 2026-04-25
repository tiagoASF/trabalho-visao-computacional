from ultralytics import YOLO

def print_metricas(model, particao):
    if particao not in ('val', 'test'):
        print('particao inválida')
        return
    
    metricas = model.val(data=path_config_yaml, verbose=False, split=particao)
    print('\n\n-----------------------')
    print('Metricas particao', particao)
    print('---------------------------')
    #Precisao média pegando 50% de coincidencia entre as imagens para cada classe
    print(f'   mAP@0.50          : {metricas.box.map50:0.3f}')
    print(f'   mAP@0.50:0.95     : {metricas.box.map:0.3f}')
    print(f'   Precisao media    : {metricas.box.mp:0.3f}')
    print(f'   Recall medio      : {metricas.box.mr:0.3f}')
    print(60*'-')
    for i, cls_idx in enumerate(metricas.box.ap_class_index):
        nome = model.names[cls_idx]
        ap   = metricas.box.ap50[i]
        print(f'  {int(cls_idx)} {nome:<15} AP@50 = {ap:.4f}')

if __name__ == '__main__':
    # instancia modelo inicial
    yolo_custom = YOLO("yolov8n.pt")
    # dados identificacao do projeto/modelo
    nome_projeto = "transfer_ep30"
    nome_modelo = "yolo_transfer"
    path_config_yaml = 'config.yaml'
    # treino
    results_treino = yolo_custom.train(
        data=path_config_yaml,
        epochs=30,
        imgsz=640,
        batch=8,
        device='mps',
        project=nome_projeto,
        name=nome_modelo,
        exist_ok=True,
        patience=15,
        plots=True, 
        amp=False,
        verbose=False
    )

    best_model_path = f'runs/detect/{nome_projeto}/{nome_modelo}/weights/best.pt'
    # instancia melhor época modelo treinado
    yolo_best = YOLO(best_model_path)    

    print_metricas(yolo_best, 'val')
    print_metricas(yolo_best, 'test')

