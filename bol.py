import cv2
import numpy as np

# nome do aluno
aluno_nome = 'Alicia'

# Variável para contar os rebotes
rebotes = 0

# Função para detectar e rastrear a bola branca e contar os rebotes
def track_and_count_bounces(video_capture):
    global rebotes  # Usamos a variável global para contar os rebotes
    while True:
        # Captura um quadro do vídeo
        ret, frame = video_capture.read()
        
        # Converte o quadro para o espaço de cores HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Defina o intervalo de tons de branco na imagem (pode precisar de ajustes)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Crie uma máscara para identificar os tons de branco na imagem
        mask = cv2.inRange(hsv_frame, lower_white, upper_white)
        
        # Aplique uma operação morfológica para eliminar ruído
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Encontre os contornos na máscara
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Inicialize o centro da bola como None
        center = None
        
        # Se houver contornos detectados
        if len(contours) > 0:
            # Encontre o maior contorno (a bola)
            c = max(contours, key=cv2.contourArea)
            
            # Encontre o centro da bola
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Calcula o momento para encontrar o centro exato
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            # Se o raio da bola for suficientemente grande, desenhe-a e rastreie
            if radius > 15:
                cv2.circle(frame, (int(x -7), int(y +7)), int(radius +8), (255, 255, 255), 2)
                cv2.circle(frame, center, 10, (0, 0, 300), -0)
                
                # Verifique se a bola cruzou a linha horizontal na parte inferior do quadro
                if center[1] + radius > frame.shape[0]:
                    rebotes += 1
                
            # Adicione o nome do aluno ao quadro na posição do centro da bola
            cv2.putText(frame, aluno_nome, center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Desenhe a linha horizontal na parte inferior do quadro
        cv2.line(frame, (0, frame.shape[0]), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
        
        # Exiba o número de rebotes no quadro
        cv2.putText(frame, f'Rebotes: {rebotes}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (-255, -255, -255), 2)
        
        # Exiba o quadro resultante
        cv2.imshow("White Ball Tracking", frame)
        
        # Saia do loop quando a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libere a captura de vídeo e feche todas as janelas
    video_capture.release()
    cv2.destroyAllWindows()

# Inicialize a captura de vídeo (0 para a câmera padrão)
cap = cv2.VideoCapture('white.avi')

# Chame a função para rastrear a bola branca e contar os rebotes
track_and_count_bounces(cap)

# Exiba o número total de rebotes ao final do rastreamento
print(f'Número total de rebotes: {rebotes}')