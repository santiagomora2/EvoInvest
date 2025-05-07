import streamlit as st
import streamlit_shadcn_ui as ui
import numpy as np
import pandas as pd
import altair as alt

st.header('EvoInvest - Aprendizaje')

tab = ui.tabs(options=['Algoritmo', 'Fórmulas', 'Prueba AG!'], default_value='Algoritmo', key="tabs_learn")

with st.sidebar:

    st.page_link("app.py", label="Regresar a Inicio", icon="🏠")

    st.page_link("pages/popt.py", label="Genera Portafolios!", icon="📈")

if tab == 'Algoritmo':

    st.markdown('''#### ¿Qué es un Algoritmo Genético?
                
Un **algoritmo genético** (AG) es un método de optimización basado en la **teoría de la evolución** de Darwin. Su objetivo es "evolucionar" hacia la **mejor solución** mediante iteraciones sucesivas. 
                
#### ¿Cómo lo hace? ¡Te cuento el proceso!

1. **Inicialización**: Creamos una “población” inicial de posibles soluciones (diferentes combinaciones de activos en el portafolio).

2. **Evaluación**: Cada “individuo” (o solución) se evalúa usando una **función de aptitud** que mide qué tan bueno es. Aquí, usamos una métrica llamada Sharpe Ratio.

3. **Selección**: Se eligen las soluciones con mayor aptitud.

4. **Cruce y Mutación**: Estas soluciones se "combinan" y "mutan" para formar nuevas soluciones, permitiendo **exploración y diversidad**.

5. **Iteración**: Repetimos este ciclo, y al pasar de generaciones, las soluciones se vuelven cada vez **mejores.**
                
#### Pero, ¿si tengo más de un objetivo?
                
En problemas de optimización de portafolio, muchas veces no basta con solo **maximizar la rentabilidad** o **minimizar el riesgo**; queremos hacer **ambas cosas al mismo tiempo**. Aquí es donde entra la optimización **multiobjetivo**.

#### ¿Qué es la Optimización Multiobjetivo?
                
La optimización multiobjetivo busca soluciones que **equilibren** dos o más objetivos **a la vez**. En este caso, queremos **maximizar los retornos y minimizar el riesgo** de nuestro portafolio. Como estos dos objetivos pueden entrar en conflicto, el algoritmo trata de encontrar un **balance óptimo**.

#### Frente de Pareto
                
Un concepto clave en la optimización de portafolios es el **frente de Pareto**. Esta es una gráfica que muestra las **mejores combinaciones** de riesgo y retorno para un conjunto de activos. Los puntos en el frente representan **portafolios “eficientes”** que ofrecen el máximo retorno posible para un nivel de riesgo dado, o el riesgo mínimo para un nivel de retorno.

#### Ejemplo de un Frente de Pareto                
''')
    
    # Generar el frente de Pareto (y = 1/x)
    pareto_x = np.linspace(0.2, 3, 20)  # Valores de x para el frente de Pareto
    pareto_y = np.exp(pareto_x)               # Valores de y para el frente
    frente_pareto = pd.DataFrame({
        "Funcion_objetivo_1": pareto_x,
        "Funcion_objetivo_2": pareto_y,
        "Tipo": ["Pareto"] * len(pareto_x)
    })

    # Generar soluciones dominadas (arriba o a la derecha del frente de Pareto)
    soluciones_dominadas_x = np.random.uniform(0.2, 3, 30)  # Más dispersas en x
    soluciones_dominadas_y =  np.exp(soluciones_dominadas_x) + np.random.uniform(3, 8, 30)  # Arriba del frente
    soluciones_dominadas = pd.DataFrame({
        "Funcion_objetivo_1": soluciones_dominadas_x,
        "Funcion_objetivo_2": soluciones_dominadas_y,
        "Tipo": ["Dominada"] * len(soluciones_dominadas_x)
    })

    # Combinar los datos
    data = pd.concat([frente_pareto, soluciones_dominadas])

    # Crear scatter plot
    chart_data = data.rename(columns={"Funcion_objetivo_1": "Función Objetivo 1", "Funcion_objetivo_2": "Función Objetivo 2"})
    scatter_plot = alt.Chart(chart_data).mark_point().encode(
        x='Función Objetivo 1',
        y='Función Objetivo 2',
        color=alt.Color('Tipo', scale=alt.Scale(domain=['Pareto', 'Dominada'], range=['#000000', '#A9A9A9'])),
    ).interactive()

    st.altair_chart(scatter_plot, use_container_width=True)

elif tab == 'Fórmulas':

    # Retornos
    st.markdown("#### Retornos: ¿Por qué son importantes?")
    st.markdown("""
    Los **retornos** representan el **cambio** relativo en el **precio de un stock** en específico en un periodo de **tiempo**.
    """)
    with st.expander("Conocer la fórmula"):
        st.markdown(r'''
        **Fórmula:**
        $\mu_i = \ln{\frac{v_t}{v_{t-1}}}$
        
        **Descripción:**
        * $\mu_i$: Retorno esperado del stock $i$.
        * $v_t$: Valor del stock en el tiempo $t$.
        * $v_{t-1}$: Valor del stock en el tiempo $t-1$.
        
        Con estos valores, se construye una matriz de retornos $\Mu$.
        ''')

    # Riesgo
    st.markdown("#### Riesgo: ¿Qué mide y por qué es crucial?")
    st.markdown("""
    El **riesgo** mide la **incertidumbre** asociada a los retornos de los stocks y la relación entre **cada par de stocks**. 
    Se calcula mediante la matriz de **covarianza**, que captura las correlaciones entre los diferentes activos.
    """)
    with st.expander("Conocer la fórmula"):
        st.markdown('''
        **Fórmula:**
        $\Sigma = \mathbb{Cov}(\Mu)$

        **Descripción:**
        * $\Mu$: Matriz de retornos.
        * $\Sigma$: Matriz de covarianza de $\Mu$.
        * $\Sigma_{i, j}$: Covarianza (riesgo conjunto) entre los stocks $i$ y $j$.
        ''')

    # Métricas del portafolio
    st.markdown("#### Métricas del portafolio: ¿Cómo optimizar retornos y riesgos?")
    st.markdown("""
    Estas métricas son clave para evaluar el **desempeño** de un **portafolio** y encontrar el **balance** ideal entre riesgo y retorno.
    Estas calculan, para cada stock, su **retorno** esperado y, para todos los stocks, el **riesgo** asociado.""")
    with st.expander("Conocer las fórmulas"):
        st.markdown('''
        **1. Retorno de inversión:**
        $R = \sum_{i=n}^{n}w_i \mu_i$

        * $R$: Retorno de inversión del portafolio.
        * $w_i$: Peso asociado al stock $i$.
        * $\mu_i$: Retorno esperado del stock $i$.

        **2. Riesgo asociado:**
        $\sigma = W \Sigma W^T$

        * $\sigma$: Desviación estándar del portafolio.
        * $W$: Vector de pesos del portafolio.
        * $\Sigma$: Matriz de covarianza de $\Mu$.

        **3. Sharpe Ratio:**
        ${(\sum_{i=n}^{n}w_i \mu_i - r)}/{\sigma}$

        * $r$: Tasa libre de riesgo (por ejemplo, $r=0.02$).
        * Otras variables como en las fórmulas anteriores.
        ''')

    # Restricciones
    st.markdown("#### Restricciones: ¿Qué condiciones aseguran un portafolio válido?")
    st.markdown("""
    Las restricciones **garantizan** que el **portafolio** cumpla condiciones básicas, como distribuir **correctamente** el **capital** entre los **activos**.
    """)
    with st.expander("Conocer las restricciones"):
        st.markdown('''
        **Restricciones:**
        * $\sum_{i=n}^{n}w_i = 1$: La suma de los pesos debe ser 1.
        * $w_i \geq 0$: Los pesos deben ser positivos (no se permite "short selling").

        Estas condiciones aseguran que el portafolio sea físicamente realizable.
        ''')

        
elif tab == 'Prueba AG!':

    st.markdown('''
#### Prueba el algoritmo genético!
                
En este **ejemplo** sencillo, introduce una palabra que el **algoritmo** deberá **encontrar**. Ajusta los **parámetros** del algoritmo y observa cómo el **algoritmo** se acerca a la palabra en cada iteración.                
''')

    st.sidebar.header("Parámetros del Algoritmo Genético")

    # Parámetros del algoritmo genético
    target_word = st.sidebar.text_input("Palabra Objetivo", "GENETICA")
    population_size = st.sidebar.slider("Tamaño de la Población", 10, 200, 50)
    num_generations = st.sidebar.slider("Número de Generaciones", 1, 500, 100)
    mutation_rate = st.sidebar.slider("Tasa de Mutación", 0.0, 1.0, 0.05)
    print_every = st.sidebar.number_input("Imprimir cada cuántas generaciones", min_value=1, value=3, step=1)

    # Configuraciones iniciales
    target_word = target_word.upper()
    word_length = len(target_word)
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")

    # Generar una población inicial aleatoria
    def generate_population(size, word_length):
        return [''.join(np.random.choice(alphabet, word_length)) for _ in range(size)]

    # Función de aptitud: mide cuántos caracteres coinciden con el objetivo
    def fitness(individual, target_word):
        return sum(1 for a, b in zip(individual, target_word) if a == b)

    # Selección de los mejores individuos
    def selection(population, target_word):
        fitness_scores = [fitness(ind, target_word) for ind in population]
        sorted_population = [ind for _, ind in sorted(zip(fitness_scores, population), reverse=True)]
        return sorted_population[:len(population) // 2]

    # Cruce: combina dos individuos
    def crossover(parent1, parent2):
        split_point = np.random.randint(1, len(parent1))
        child = parent1[:split_point] + parent2[split_point:]
        return child

    # Mutación: cambia un carácter aleatoriamente en el individuo
    def mutate(individual, mutation_rate):
        individual = list(individual)
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                individual[i] = np.random.choice(alphabet)
        return ''.join(individual)

    # Evolución de la población
    def evolve_population(population, target_word, mutation_rate):
        selected = selection(population, target_word)
        children = []
        while len(children) < len(population):
            parent1, parent2 = np.random.choice(selected, 2, replace=False)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            children.append(child)
        return children

    # Botón para iniciar la simulación
    if st.button("Ejecutar Algoritmo Genético"):

        gens = st.expander('Ver evolución')

        # Generación inicial
        population = generate_population(population_size, word_length)
        best_match = ""
        for generation in range(num_generations):
            # Evolución de la población
            population = evolve_population(population, target_word, mutation_rate)
            best_match = max(population, key=lambda ind: fitness(ind, target_word))
            
            # Imprimir cada N generaciones
            with gens:
                if (generation + 1) % print_every == 0:
                    st.write(f"**Generación {generation + 1}:** Mejor combinación encontrada: `{best_match}` - Aptitud: {fitness(best_match, target_word)}")

            # Condición de parada si encontramos la palabra objetivo
            if best_match == target_word:
                st.success(f"¡La palabra objetivo '{target_word}' fue encontrada en la generación {generation + 1}!")
                break
        else:
            st.warning("El algoritmo no logró encontrar la palabra objetivo en el número de generaciones especificado.")

        # Mostrar el mejor resultado final si no se alcanzó el objetivo
        if best_match != target_word:
            st.write(f"**Mejor combinación final** después de {num_generations} generaciones: `{best_match}` - Aptitud: {fitness(best_match, target_word)}")

        st.markdown(f'''
#### ¿Qué es lo que está pasando?
                    
1. Se **generan {population_size}** palabras **(individuos)** de {len(target_word)} caracteres, con letras **aleatorias**.

2. De estos individuos, se **evalúan** cuántas letras comparten con {target_word}, y ese es el **valor de fitness** de cada palabra.

3. Se **seleccionan** los individuos con base en su **fitness** y, como en la naturaleza, sobreviven los **más aptos**.

4. Se **reproducen** con cierta probabilidad y los nuevos individuos tienen una pequeña probabilidad de **mutar** uno de sus **genes** (letras de la palabra).

5. Se **repite** este proceso desde el paso 2, con cada **nueva población**, hasta obtener el **mayor fitness** o, en este caso, llegar a la palabra {target_word}.
''')
