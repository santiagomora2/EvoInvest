import streamlit as st
import streamlit_shadcn_ui as ui
import numpy as np

st.header('EvoInvest - Aprendizaje')

tab = ui.tabs(options=['Algoritmo Genético', 'Conceptos Financieros / Fórmulas', 'Prueba Algoritmo Genético'], default_value='Algoritmo Genético', key="tabs_learn")

with st.sidebar:

    st.page_link("app.py", label="Regresar a Inicio", icon="🏠")

    st.page_link("pages/popt.py", label="Genera Portafolios!", icon="📈")

if tab == 'Algoritmo Genético':

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
''')

elif tab == 'Conceptos Financieros / Fórmulas':

    st.markdown('''
#### Retornos
                
Los retornos de cada stock se calculan de la siguiente manera:
                
$\mu_i = \ln{(v_t - v_{t-1})}$
                
* Donde:
    * $\mu_i$: Retorno esperado del stock $i$
    * $v_t$: Valor del stock en el tiempo $t$
    * $v_{t-1}$: Valor del stock en el tiempo $t-1$

Con los retornos de cada stock en cada día, se define una matriz de retornos $\Mu$.
                
#### Riesgo
                
El riesgo entre dos stocks diferentes se obtiene calculando la matriz de covarianza:
                
$\Sigma = \mathbb{Cov}(\Mu)$
                
* Donde:
    * $M$: matriz de retornos
    * $\Sigma$: Matriz de covarianza de $M$
    * $\Sigma_{i, j}$: covarianza (riesgo) entre el stock $i$ y el $j$
                
#### Métricas del portafolio
                
Una vez definidos los retornos y riesgos de cada stock, se usa el AG para optimizar los pesos de cada stock en el portafolio, utilizando las siguientes métricas:
                
##### Retorno de inversión:
                
$ R = \sum_{i=n}^{n}w_i \mu_i$

* Donde:
    * $R$: Retorno de inversión del portafolio
    * $w_i$: Peso asociado al stock $i$
    * $\mu_i$: Retorno esperado del stock $i$

##### Riesgo asociado:

$ \sigma = W \Sigma W^T$

* Donde:
    * $\sigma$: Desviación estándar asociada al portafolio
    * $W$: Vector de pesos del portafolio
    * $\Sigma$: Matriz de covarianza de $\Mu$
                
##### Sharpe Ratio
                
$ {(\sum_{i=n}^{n}w_i \mu_i - r)}/{\sigma}$

* Donde:
    * $w_i$: Peso asociado al stock $i$
    * $\mu_i$: Retorno esperado del stock $i$
    * $r$: risk free rate (generalmente $r=0.02$)
    * $\sigma$ Desviación estándar asociada al portafolio
                
### Restricciones
                
Para optimizar los pesos de cada stock, mediante el Algoritmo Genético, definimos las siguientes restricciones:

* $\sum_{i=n}^{n}w_i = 1$
* $w_i \geq 0, i = 1, 2, \dots, n $

* Donde:
    * $w_i$: Peso asociado al stock $i$
                ''')
    
elif tab == 'Prueba Algoritmo Genético':

    st.markdown('''
#### Prueba el algoritmo genético!
                
En este **ejemplo** sencillo, introduce una palabra que el **algoritmo** deberá **encontrar**. Ajusta los **parámetros** del algoritmo y observa cómo el **algoritmo** se acerca a la palabra en cada iteración.

En este caso, la **función de fitness** es el número de letras que cada individuo comparte con la **palabra objetivo**.           
                
''')

    st.sidebar.header("Parámetros del Algoritmo Genético")

    # Parámetros del algoritmo genético
    target_word = st.sidebar.text_input("Palabra Objetivo", "GENETICA")
    population_size = st.sidebar.slider("Tamaño de la Población", 10, 200, 100)
    num_generations = st.sidebar.slider("Número de Generaciones", 1, 500, 100)
    mutation_rate = st.sidebar.slider("Tasa de Mutación", 0.0, 1.0, 0.05)
    print_every = st.sidebar.number_input("Imprimir cada cuántas generaciones", min_value=1, value=5, step=1)

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
        # Generación inicial
        population = generate_population(population_size, word_length)
        best_match = ""
        for generation in range(num_generations):
            # Evolución de la población
            population = evolve_population(population, target_word, mutation_rate)
            best_match = max(population, key=lambda ind: fitness(ind, target_word))
            
            # Imprimir cada N generaciones
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
