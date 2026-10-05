"""
Actividad 29: Análisis de Sensibilidad a la Recompensa en Q-Learning
Maestría en Inteligencia Artificial - TecNM / ITSU
Materia: Aprendizaje Automático
Autor: Juan Pablo Figueroa Moran (M26040059)
Repositorio: https://github.com/juanpablofm92/Actividad29.git
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class BranchingMDP:
    """
    Proceso de Decisión de Markov con Bifurcación:
    - S1: Nodo raíz de bifurcación (Inicio). Acciones: A1 (ir a S2), A2 (ir a S3).
    - S2: Ruta de recompensa intermedia fija: r(S2) = +4.0.
    - S3: Ruta de recompensa de prueba: r(S3) varía según el experimento.
    """
    def __init__(self, r_s3: float):
        self.r_s3 = r_s3
        self.states = ["S1", "S2", "S3"]
        self.actions = ["A1", "A2"]

    def step(self, state: str, action: str):
        if state == "S1":
            if action == "A1":
                return "S2", 0.0, False
            else:
                return "S3", 0.0, False
        elif state == "S2":
            if action == "A1":
                return "TERMINAL", 4.0, True
            else:
                return "S2", -0.5, False
        elif state == "S3":
            if action == "A2":
                return "TERMINAL", self.r_s3, True
            else:
                return "S3", -0.5, False
        return "TERMINAL", 0.0, True


def train_q_learning(r_s3: float, episodes: int = 500, alpha: float = 0.2, gamma: float = 0.9):
    mdp = BranchingMDP(r_s3)
    states = mdp.states
    actions = mdp.actions
    
    Q = {s: {a: 0.0 for a in actions} for s in states}
    epsilon = 0.3

    for ep in range(episodes):
        s = "S1"
        done = False
        curr_eps = max(0.01, epsilon * (0.995 ** ep))

        while not done:
            # Política epsilon-greedy
            if np.random.rand() < curr_eps:
                a = np.random.choice(actions)
            else:
                # Argmax con desempate aleatorio
                max_v = max(Q[s].values())
                best_actions = [act for act, v in Q[s].items() if np.isclose(v, max_v)]
                a = np.random.choice(best_actions)

            s_prime, r, done = mdp.step(s, a)

            max_q_next = 0.0 if done else max(Q[s_prime].values())
            td_target = r + gamma * max_q_next
            Q[s][a] += alpha * (td_target - Q[s][a])

            s = s_prime

    # Convertir a DataFrame ordenado
    df = pd.DataFrame(Q).T[actions]
    return df


def extract_policy(df_q: pd.DataFrame) -> dict:
    policy = {}
    for state in df_q.index:
        best_a = df_q.loc[state].idxmax()
        policy[state] = best_a
    return policy


def main():
    print("*" * 80)
    print(" TECNM / INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN - MAESTRÍA EN IA")
    print(" ACTIVIDAD 29: ANÁLISIS DE SENSIBILIDAD A LA RECOMPENSA EN Q-LEARNING")
    print(" Alumno: Juan Pablo Figueroa Moran | Matrícula: M26040059")
    print("*" * 80)

    # 1. Ejecución de los experimentos
    print("\n[+] Entrenando agente bajo Experimento A: Recompensa Base r(S3) = +3.0 ...")
    df_qa = train_q_learning(r_s3=3.0, episodes=800)

    print("[+] Entrenando agente bajo Experimento B: Recompensa Modificada r(S3) = +6.0 ...")
    df_qb = train_q_learning(r_s3=6.0, episodes=800)

    # 2. Cálculo de la matriz de diferencia delta_Q = Q_B - Q_A
    df_delta = df_qb - df_qa

    # 3. Presentación en consola de resultados
    print("\n" + "=" * 65)
    print(" TABLA Q - EXPERIMENTO A: Recompensa Base r(S3) = +3.0")
    print("=" * 65)
    print(df_qa.round(4).to_string())
    policy_a = extract_policy(df_qa)
    print(f"-> Política Óptima en S1: {policy_a['S1']} (Prioriza Rama S2 con r=+4.0 > +3.0)")

    print("\n" + "=" * 65)
    print(" TABLA Q - EXPERIMENTO B: Recompensa Modificada r(S3) = +6.0")
    print("=" * 65)
    print(df_qb.round(4).to_string())
    policy_b = extract_policy(df_qb)
    print(f"-> Política Óptima en S1: {policy_b['S1']} (Reorientación hacia Rama S3 con r=+6.0 > +4.0)")

    print("\n" + "=" * 65)
    print(" MATRIZ DE DIFERENCIA DELTA_Q = Q_B - Q_A")
    print("=" * 65)
    print(df_delta.round(4).to_string())

    print("\n" + "=" * 65)
    print(" ANÁLISIS DE LA REORIENTACIÓN DE LA CONDUCTA:")
    print("=" * 65)
    print(f" * Decisión en Bifurcación S1:")
    print(f"   - Experimento A: Q(S1, A1) = {df_qa.loc['S1', 'A1']:.2f} > Q(S1, A2) = {df_qa.loc['S1', 'A2']:.2f} => Acción elegida: A1 (Ruta S2)")
    print(f"   - Experimento B: Q(S1, A2) = {df_qb.loc['S1', 'A2']:.2f} > Q(S1, A1) = {df_qb.loc['S1', 'A1']:.2f} => Acción elegida: A2 (Ruta S3)")
    print(f"   - Impacto Delta: Delta_Q(S1, A2) = +{df_delta.loc['S1', 'A2']:.2f} (Incentivo propagado por Bellman)")
    print(f"   - Conclusión: El incremento de valor en S3 alteró radicalmente la trayectoria óptima.")

    # 4. Generación de mapas de calor con Seaborn
    os.makedirs("docs", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(df_qa, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=axes[0], annot_kws={"size": 13, "weight": "bold"})
    axes[0].set_title(r"Tabla $Q_A$ (Recompensa Base $r(S_3)=+3.0$)" + "\n" + r"Política Óptima $\pi^*(S_1) = A_1$", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Estados")
    axes[0].set_xlabel("Acciones")

    sns.heatmap(df_qb, annot=True, fmt=".2f", cmap="YlOrBr", cbar=True, ax=axes[1], annot_kws={"size": 13, "weight": "bold"})
    axes[1].set_title(r"Tabla $Q_B$ (Recompensa Modificada $r(S_3)=+6.0$)" + "\n" + r"Política Óptima $\pi^*(S_1) = A_2$", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Estados")
    axes[1].set_xlabel("Acciones")

    sns.heatmap(df_delta, annot=True, fmt=".2f", cmap="magma", cbar=True, ax=axes[2], annot_kws={"size": 13, "weight": "bold"})
    axes[2].set_title(r"Matriz de Sensibilidad $\Delta Q = Q_B - Q_A$" + "\nReorientación de Preferencia hacia $S_3$", fontsize=11, fontweight='bold')
    axes[2].set_ylabel("Estados")
    axes[2].set_xlabel("Acciones")

    plt.tight_layout()
    output_path = os.path.join("docs", "mapa_calor_delta_q.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[+] Mapa de calor exportado exitosamente en: {output_path}")


if __name__ == "__main__":
    main()
