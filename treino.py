from ultralytics import YOLO

def print_metricas(model, particao, path_config_yaml):
    if particao not in ('val', 'test'):
        print('particao invalida')
        return
    
    metricas = model.val(data=path_config_yaml, verbose=False, split=particao)
    
    print('\n' + '='*70)
    print(f'  Métricas — partição: {particao}')
    print('='*70)
    
    # Métricas agregadas (médias entre classes)
    print(f'  mAP@0.50              : {metricas.box.map50:.3f}')
    print(f'  mAP@0.50:0.95         : {metricas.box.map:.3f}')
    print(f'  Precisão média        : {metricas.box.mp:.3f}')
    print(f'  Recall médio          : {metricas.box.mr:.3f}')
    
    # Métricas por classe
    print('-'*70)
    print(f'  {"Classe":<15} {"P":>8} {"R":>8} {"AP@50":>10} {"AP@50:95":>10}')
    print('-'*70)
    for i, cls_idx in enumerate(metricas.box.ap_class_index):
        nome = model.names[cls_idx]
        p     = metricas.box.p[i]       # precisão da classe
        r     = metricas.box.r[i]       # recall da classe
        ap50  = metricas.box.ap50[i]    # AP@50 da classe
        ap    = metricas.box.ap[i]      # AP@50:95 da classe (média de IoU 0.5 a 0.95)
        f1    = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        print(f'  {nome:<15} {p:>8.3f} {r:>8.3f} {ap50:>10.3f} {ap:>10.3f}   F1={f1:.3f}')
    print('='*70 + '\n')

if __name__ == '__main__':
    # instancia modelo inicial
    yolo_custom = YOLO("yolov8n.pt")
    # dados identificacao do projeto/modelo
    nome_projeto = "transfer_ep100"
    nome_modelo = "yolo_transfer"
    path_config_yaml = 'config.yaml'
    # treino
    results_treino = yolo_custom.train(
        data=path_config_yaml,
        epochs=100,
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

   # avaliação nas duas partições
    print_metricas(yolo_best, 'val', path_config_yaml)
    print_metricas(yolo_best, 'test', path_config_yaml)

