import os
import shutil
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
# ---------------------------------------------------------------------------

# insira seu código aqui

# Dica de estrutura (não é obrigatório seguir exatamente assim):
#
# model = YOLO("model.pt")
# model.export(format="tflite", imgsz=...)

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_dir, "model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    print("Carregando modelo treinado...")
    model = YOLO(model_path)

    print("Exportando para TFLite...")
    model.export(format="tflite", imgsz=640, int8=True)

    # o export gera vários arquivos (Float16, Float32, Int8)
    # então precisa procurar especificamente o int8
    int8_model = None
    for root, _, files in os.walk(project_dir):
        for f in files:
            if f.endswith(".tflite") and "int8" in f.lower():
                int8_model = os.path.join(root, f)
                break
        if int8_model:
            break

    if int8_model is None:
        raise FileNotFoundError(".tflite int8 gerado pelo export não localizado.")

    output_model = os.path.join(project_dir, "model.tflite")
    shutil.copy(int8_model, output_model)

    print(f"Exportação concluída. Modelo salvo em: {output_model}")


if __name__ == "__main__":
    main()
