# Presentación y guía de defensa

**Proyecto:** Candidatos a cupones personalizados — Proyecto Integrador AIAI
**Equipo:** Alejandra Ramírez, Katerin Gonzalez, Luisa Flores, Alvaro Barreto
**Formato:** unos 10 minutos de exposición con demo, y después preguntas técnicas
individuales.

Este documento tiene tres partes:

1. **Diapositivas:** contenido de cada una, notas para quien habla y tiempo.
2. **Guion de la demo:** qué mostrar en el prototipo, en orden, con plan B.
3. **Guía de defensa:** preguntas probables con respuestas cortas. Cualquier
   integrante debe poder responder cualquiera.

> Las cifras salen de `INFORME_FINAL.md` y `BITACORA_DECISIONES.md`. Si algo se
> regenera, hay que actualizarlas.

---

# Parte 1 — Diapositivas

Reparto sugerido: cada integrante presenta un bloque, pero **todos deben poder
explicar todo**, porque la defensa es individual.

| Bloque | Diapositivas | Tiempo | Sugerido |
| --- | --- | ---: | --- |
| Problema y datos | 1–3 | 1:45 | Integrante 1 |
| Método | 4–6 | 2:30 | Integrante 2 |
| Resultados | 7–9 | 2:15 | Integrante 3 |
| Demo y cierre | 10–12 | 3:30 | Integrante 4 |
| **Total** | 12 | **10:00** | |

---

## Diapositiva 1 — Título (0:15)

**Candidatos a cupones personalizados**
Un recomendador repeat + explore a partir del historial de compras

ITRN · Aplicaciones de Aprendizaje Automático · Proyecto Integrador
Alejandra Ramírez · Katerin Gonzalez · Luisa Flores · Alvaro Barreto

> **Notas:** presentarse en una frase y pasar directo al problema.

---

## Diapositiva 2 — El problema (0:45)

**Una misma promoción no le sirve a todos**

- Los supermercados lanzan promociones generales.
- Con un programa de fidelización, cada compra queda asociada a un hogar.
- **Pregunta:** ¿cómo usar ese historial para elegir sobre qué productos
  ofrecerle un cupón a cada cliente?

**Usuario:** el área comercial o de fidelización.
**Salida:** 10 productos candidatos por hogar, cada uno con su explicación.

> **Notas:** aclarar que el sistema **propone candidatos**; no decide si se
> emite un cupón. Esa decisión pasa por presupuesto, stock y margen.

---

## Diapositiva 3 — Datos (0:45)

**dunnhumby — The Complete Journey**

| | |
| --- | ---: |
| Líneas de compra | 2,6 millones |
| Hogares | 2.500 |
| Canastas | 276.484 |
| Período | 711 días |

- Calidad: 0 nulos, 0 duplicados, 0 productos sin catálogo.
- Se descartan 14.466 líneas con cantidad cero, sin borrar los originales.
- **No** usamos datos demográficos: el foco es el comportamiento y evitamos
  variables sensibles.

> **Notas:** si sobra tiempo, mencionar que el pipeline verifica con hashes que
> los datos originales nunca cambian.

---

## Diapositiva 4 — Dos objetivos: repeat y explore (0:50)

**La mitad de cada canasta es recompra**

*(gráfico: `outputs/eda/defensa_repeticion_por_canasta.png`)*

- Repeat: productos que el hogar **ya compra**. El 50 % de cada canasta es
  recompra.
- Explore: productos **nuevos** para el hogar. Es la otra mitad, y mucho más
  difícil de predecir.
- Si se evalúan juntos, el sistema se optimiza hacia recompras y explore
  desaparece.

> **Notas:** la recompra crece con la antigüedad del cliente: 12 % en las
> primeras canastas y 64 % después de la canasta 100. Esto justifica medir cada
> objetivo por separado.

---

## Diapositiva 5 — Cómo evitamos hacer trampa (0:50)

**División temporal por hogar**

```text
Hogar:  [ c1  c2  c3  …  c(n-2) ] [ c(n-1) ] [ c(n) ]
          entrenamiento             validación  prueba
                                    elegir      medir UNA vez
```

- Se predice el futuro con el pasado: nunca se usan compras posteriores.
- **Validación** sirve para comparar modelos y elegir parámetros.
- **Prueba** se abrió **una sola vez**, con todo ya congelado.
- La división es por hogar porque los hogares entran al dataset en distintos
  momentos.

> **Notas:** este es el punto metodológico central. Decir "fuga de
> información" y "prueba sellada".

---

## Diapositiva 6 — Modelos (0:50)

**Probamos de lo simple a lo complejo**

| Objetivo | Modelos evaluados | Elegido |
| --- | --- | --- |
| Repeat | Frecuencia + recencia · ALS | **Frecuencia + recencia** |
| Explore | Popularidad · Coocurrencia · Híbrido · ALS | **ALS** |

- **Frecuencia + recencia:** `0,7 × frecuencia + 0,3 × recencia`.
- **ALS implícito:** aprende "gustos latentes" de hogares y productos a partir
  de todo el historial.
- Implementamos ALS **a mano** (numpy y scipy), porque Windows bloqueaba la
  librería. Una prueba verifica que coincide con la solución exacta.

> **Notas:** remarcar que **no elegimos por sofisticación**. En repeat ganó el
> modelo más simple.

---

## Diapositiva 7 — Resultados en validación (0:45)

**Hit Rate@10:** % de hogares con al menos un acierto entre los 10 sugeridos.

| | Repeat | Explore |
| --- | ---: | ---: |
| Frecuencia + recencia | **64,5 %** | — |
| ALS | 46,2 % | **9,5 %** |
| Popularidad | — | 7,2 % |
| Coocurrencia / Híbrido | — | 5,2 % / 5,9 % |

- ALS recomienda **1.051 productos distintos**; popularidad, solo 81.
- La coocurrencia se quedó sin memoria con umbrales bajos, y al subirlos perdió
  cobertura.

> **Notas:** explore es mucho más difícil. 9,5 % no es poco: hay unos 90.000
> productos posibles.

---

## Diapositiva 8 — ¿Cuántos repeat y cuántos explore? (0:45)

| Lista de 10 | Hit Rate total | Aciertan algo nuevo |
| --- | ---: | ---: |
| 10 + 0 | 52,7 % | 0 % |
| **7 + 3** | **50,1 %** | **4,0 %** |
| 5 + 5 | 46,4 % | 6,1 % |

- Con más repeat, la métrica siempre sube: si solo miráramos la métrica,
  eliminaríamos explore.
- Decidimos con una **regla de producto**: al menos 3 productos nuevos. Cuesta
  2,6 puntos.

> **Notas:** la métrica sola no alcanza para decidir, y esta diapositiva lo
> muestra.

---

## Diapositiva 9 — Resultado final sobre prueba (0:45)

**Evaluación única, con datos nunca vistos**

| | Prueba | Validación |
| --- | ---: | ---: |
| **Sistema 7 + 3** | **50,7 %** | 50,1 % |
| Repeat | 64,5 % | 64,5 % |
| Explore ALS | 8,6 % | 9,5 % |
| Explore popularidad (referencia) | 6,5 % | 7,2 % |

- Es estable: prueba ≈ validación.
- Explore baja un poco, pero popularidad también, así que la caída se explica
  por la canasta de prueba.
- **La mitad de los hogares compró al menos uno de los 10 productos sugeridos.**

> **Notas:** mide coincidencia con compras, **no** que el cupón cause la compra.

---

## Diapositiva 10 — Reglas de negocio (0:45)

**Relevante no es lo mismo que ofrecible**

| Regla | Por qué |
| --- | --- |
| Sin combustible | Aparecía en la lista del 45 % de los hogares. |
| Sin tabaco | Promoción prohibida (Ley 18.256). |
| Sin variantes repetidas | La misma leche con otro código no es un producto nuevo. |

- Se aplican **después** del modelo, y cada exclusión queda con su motivo.
- El Hit Rate baja (de 50 % a 45 %) porque combustible y tabaco eran aciertos
  fáciles pero no ofrecibles.
- **Las reglas de tabaco y variantes las encontramos mirando el prototipo, no
  las métricas.**

> **Notas:** esta es una lección del proyecto. La revisión cualitativa
> encuentra problemas que las métricas no muestran.

---

## Diapositiva 11 — Demo (2:00)

**Prototipo para el área comercial** — `streamlit run app/app.py`

Entrada → historial → modelo → reglas → 10 candidatos explicados → feedback

> Ver el guion en la Parte 2.

---

## Diapositiva 12 — Conclusiones y límites (0:45)

**Lo que logramos**
- Una prueba de concepto completa, reproducible (un solo comando) y con 49
  pruebas automáticas.
- 50,7 % de los hogares con al menos un acierto, y cada producto con su
  explicación.

**Lo que aprendimos**
- Un baseline simple puede ser la mejor decisión.
- La métrica no alcanza: la mezcla y las reglas son decisiones de producto.

**Lo que NO podemos afirmar**
- Que los cupones aumenten las ventas. Eso requiere una **prueba A/B**.

**Próximo paso:** más explore para clientes nuevos (ahí rinde 15,6 % frente a
3,3 %).

> **Notas:** cerrar con la frase del límite causal. Muestra criterio y suele
> evitar la pregunta.

---

# Parte 2 — Guion de la demo (2 minutos)

**Antes de presentar:**

```powershell
python -m src.preparar_demo      # solo si se cambió algo; los datos ya están generados
streamlit run app/app.py
```

- Abrir http://localhost:8501 **antes** de empezar. La primera carga tarda unos
  segundos.
- Tener abierta una captura de cada pantalla, como plan B si algo falla.
- Borrar `outputs/demo/feedback_simulado.csv` si quedaron pruebas anteriores.

**Recorrido:**

| Tiempo | Acción | Qué decir |
| ---: | --- | --- |
| 0:00 | Elegir **hogar 20** en la barra lateral. | "Elegimos un hogar con 387 compras." |
| 0:10 | Mostrar las métricas del historial. | "Esto es lo que el sistema sabe del cliente." |
| 0:20 | Pestaña **Recomendaciones**, tabla repeat. | "7 productos que ya compra. Cada uno dice por qué: lo compró 86 veces, la última en su penúltima compra." |
| 0:45 | Tabla explore. | "3 productos nuevos. Hot dog buns, porque los hogares que compran pan de hamburguesa, como este, también lo eligen." |
| 1:05 | Pestaña **Filtrados por reglas**. | "Esto lo descartamos: estaban los cigarrillos y la nafta que este hogar compra, y variantes de la misma leche. El modelo los consideraba relevantes, pero no se pueden ofrecer." |
| 1:25 | Elegir **Otro hogar** y escribir 99999. | "Un cliente sin historial recibe una lista de respaldo con productos populares elegibles." |
| 1:40 | Pestaña **Feedback** de un hogar y clic en 👍. | "Acá se registraría el canje o el rechazo del cupón, para reentrenar en el futuro." |
| 1:55 | Vista **Sobre el modelo**. | "Y los resultados medidos, con la aclaración de que no son efecto causal." |

**Si sobra tiempo:** el hogar **186** (3 canastas) muestra el aviso de historial
corto.

**Si algo falla:** decir "les muestro la captura" y seguir. No depurar en vivo.

---

# Parte 3 — Guía de defensa

Cada respuesta tiene una **versión corta** (lo que se dice primero) y, si hace
falta, un **detalle** para cuando repregunten. Dónde está cada cosa:
`INFORME_FINAL.md` (cifras), `BITACORA_DECISIONES.md` (decisiones D-001 a D-011)
y `src/` (código).

## A. Problema y datos

**¿Por qué este problema es de aprendizaje automático y no de reglas fijas?**
Porque hay 2.500 hogares y 90.000 productos: no se pueden escribir reglas a mano
para cada combinación. Los patrones de recompra y de afinidad se aprenden de los
datos. Las reglas fijas las usamos solo para la elegibilidad comercial.

**¿Por qué no usaron los datos demográficos?**
El objetivo era predecir con el comportamiento de compra. Lo demográfico es
sensible, solo existe para una parte de los hogares y podría introducir sesgos
sin una justificación clara. Queda como posible mejora, evaluándola con cuidado.

**¿Qué hicieron con las cantidades en cero?**
Son 14.466 líneas. Las excluimos de las interacciones, porque una compra
requiere unidades positivas, pero no las borramos de los originales. Está
registrado en D-002.

**¿Los resultados valen para un supermercado uruguayo?**
No necesariamente. El dataset es de un minorista estadounidense y de hace años.
El método se puede trasladar, pero los números habría que volver a medirlos con
datos locales.

## B. Metodología y fuga de información

**¿Por qué una división temporal y no aleatoria?**
Porque en producción siempre se predice el futuro con el pasado. Una división
aleatoria mezclaría compras futuras en el entrenamiento, que es fuga de
información, y daría resultados optimistas.

**¿Por qué por hogar y no un corte de fecha global?**
Los hogares entran al dataset gradualmente (hay una rampa en las primeras
semanas). Con un corte global, algunos hogares no tendrían historial para
entrenar y otros no tendrían canasta para evaluar.

**¿Qué diferencia hay entre validación y prueba?**
Validación sirve para elegir modelos, parámetros y la mezcla. Prueba se usa una
sola vez, al final, con todo congelado. Así el número final no está sesgado por
nuestras elecciones.

**¿Cómo garantizan que la prueba se usó una sola vez?**
- La configuración se congeló en `config/modelo_final.json` antes de abrir la
  prueba.
- El script de prueba se niega a sobrescribir su reporte sin `--forzar`.
- El pipeline no ejecuta esa etapa por defecto.
- Después de la prueba no cambiamos ningún modelo.

**¿Las reglas de negocio no son un cambio posterior a la prueba?**
Las reglas no cambian el modelo: filtran su salida. Su efecto lo medimos en
validación, no en prueba, y lo reportamos aparte (D-008).

## C. Modelos

**Explicá el puntaje repeat.**
`0,7 × frecuencia normalizada + 0,3 × recencia`.
- La frecuencia son las canastas que contienen el producto, divididas por las
  del producto más comprado por ese hogar.
- La recencia vale `1 / (1 + canastas desde la última compra)`.

Los pesos fueron una decisión inicial razonable. No los ajustamos para no
sobreajustar a la validación.

**¿Por qué cuentan canastas y no unidades?**
Las unidades no son comparables entre departamentos: un litro de combustible no
equivale a un yogur. Contar en cuántas canastas aparece un producto es una señal
homogénea.

**¿Qué es ALS?**
Factoriza la matriz hogar × producto en dos matrices chicas: un vector de 32
números por hogar y otro por producto. El producto punto entre ambos estima la
afinidad. Es "implícito" porque no hay calificaciones, solo compras: una compra
repetida da más confianza (`1 + 2·log(1 + canastas)`). Se llama "alternante"
porque fija los productos y resuelve los hogares, después al revés, y repite.

**¿Por qué lo implementaron ustedes?**
El Control de aplicaciones de Windows bloqueaba las librerías compiladas
(`implicit` y scikit-learn). No cambiamos esa configuración de seguridad. Lo
implementamos con numpy y scipy usando gradiente conjugado, el mismo método que
usa la librería, y una prueba automática compara el resultado con la solución
exacta.

**¿Cómo eligieron los parámetros de ALS?**
Con una grilla pequeña fijada antes de ver resultados: 2 × 2 × 2 × 2 = 16
configuraciones, evaluadas solo en validación por NDCG@10 explore. Ganó la más
simple: 32 factores y alpha 2.

**¿Por qué ALS no se usa también para repeat?**
Porque rindió mucho peor: Hit Rate 46 % contra 64,5 % del baseline. Para
recompras, la frecuencia y la recencia del propio hogar son señales más
directas que los gustos latentes.

**¿Por qué falló la coocurrencia?**
Por escala. Con un umbral de 20 canastas había 25 millones de pares y se agotó
la memoria. Con 500 canastas funcionó, pero solo sobre 573 productos, así que
perdió los de nicho. Además, el lift favorece asociaciones específicas que no
siempre son las compras más probables.

## D. Métricas y resultados

**Definí Hit Rate, Precision, Recall y NDCG.**
- **Hit Rate@10:** proporción de hogares con al menos un acierto en los 10.
- **Precision@10:** aciertos / 10.
- **Recall@10:** aciertos / productos relevantes que compró.
- **NDCG@10:** como Recall, pero un acierto en la posición 1 vale más que uno en
  la 10.

Todas se promedian por hogar.

**¿Por qué promediar por hogar?**
Para que los hogares que compran mucho no dominen el resultado. Cada cliente
pesa igual.

**¿Un Hit Rate explore del 9 % no es muy bajo?**
Es la tarea más difícil: adivinar uno de unos 90.000 productos que el hogar
nunca compró. Lo importante es la comparación con la referencia: ALS supera a
popularidad en un 32 % y la mejora se mantuvo en prueba.

**¿Por qué bajó explore en prueba?**
Bajó de 9,5 % a 8,6 %, pero popularidad, que nunca elegimos, también bajó (de
7,2 % a 6,5 %). Eso indica que la canasta de prueba era más difícil, no que
hayamos sobreajustado. La ventaja relativa de ALS se mantuvo.

**¿Por qué 7 + 3 y no lo que maximiza la métrica?**
Porque la métrica siempre mejora con más repeat; optimizarla eliminaría explore,
que es parte del objetivo. Fijamos una regla de producto (al menos 3 productos
nuevos) y, dentro de ella, elegimos el mejor NDCG. Cuesta 2,6 puntos de Hit
Rate.

**¿Qué pasa con un cliente nuevo?**
- Con menos de 3 canastas no se modela, y la app le muestra una lista de
  respaldo con productos populares elegibles.
- Con poco historial, repeat sigue funcionando (67 %), y explore rinde mejor
  que con mucho historial (15,6 % contra 3,3 %), porque el cliente todavía no
  probó lo común.

## E. Reglas, prototipo e ingeniería

**¿Por qué el Hit Rate baja con las reglas si el sistema "mejora"?**
Porque combustible y tabaco se recompran muchísimo, así que eran aciertos
fáciles pero no ofrecibles. Si se mide solo contra compras elegibles, el Hit
Rate queda en 48 %. El resto de la diferencia viene de que la métrica cuenta
aciertos por código de producto, y ahora no repetimos variantes.

**¿Cómo descubrieron lo del tabaco?**
Mirando el prototipo con datos reales: el hogar 20 recibía cigarrillos. Ninguna
métrica lo mostraba. Por eso decimos que la revisión cualitativa es parte de la
evaluación.

**¿Cómo explican cada recomendación?**
- **Repeat:** cuántas veces lo compró y cuándo fue la última vez.
- **Explore:** qué producto del historial del hogar se le parece más en el
  espacio latente de ALS.

**¿Cómo se integraría el feedback?**
En la app, 👍/👎 se registran en un CSV. En producción, los canjes y rechazos
reales se sumarían como señal y el modelo se reentrenaría periódicamente.
Reentrenar lleva unos 80 segundos.

**¿Está listo para producción?**
No. Faltan stock, margen y campañas reales (están previstos como puntos de
integración), feedback real, monitoreo y, sobre todo, una prueba A/B que mida el
efecto en ventas.

**¿Cómo se reproduce todo?**
`python -m src.pipeline` ejecuta todas las etapas y registra versiones y
tiempos. Verifica con hashes que los datos no cambien. Hay 49 pruebas, incluida
una de integración con datos sintéticos que encontró un error real.

## F. Preguntas trampa

**"Entonces, ¿cuánto aumentan las ventas con su sistema?"**
No lo sabemos, y el dataset no permite saberlo. Medimos si el cliente compró lo
sugerido, no si lo compró *por* el cupón. Para eso hace falta una prueba A/B:
un grupo con cupones personalizados y otro sin ellos.

**"Si repeat acierta tanto, ¿para qué ofrecer cupones de algo que igual van a comprar?"**
Es una muy buena objeción de negocio. Un cupón repeat puede fidelizar o
adelantar la compra, pero también puede regalar margen. Justamente por eso
existe explore, y la decisión final la toma el área comercial. Medir el efecto
incremental es la prueba A/B de las mejoras futuras.

**"¿No hicieron trampa ajustando cosas después de ver la prueba?"**
No. Después de la prueba solo agregamos reglas de elegibilidad, que no cambian
el modelo, y medimos su efecto en validación. La lista que se evaluó en prueba
la regeneramos y es idéntica a la guardada.

**"¿Por qué no usaron deep learning?"**
Porque no lo justificaban ni los datos ni los resultados. Un modelo simple ganó
en repeat y ALS es un estándar sólido para feedback implícito. Elegir por
sofisticación habría sido un error metodológico.

---

# Checklist antes de presentar

- [ ] Cada integrante leyó `INFORME_FINAL.md` completo y la Parte 3 de este
      documento.
- [ ] Ensayo cronometrado: 10 minutos o menos, incluida la demo.
- [ ] Ronda de preguntas cruzadas: cada integrante responde preguntas de un
      bloque que **no** presentó.
- [ ] App probada en el equipo de la presentación, con capturas como plan B.
- [ ] `feedback_simulado.csv` limpio.
- [ ] Gráficos exportados para las diapositivas 4 y 7 desde `outputs/eda/`.
- [ ] Cifras clave memorizadas: **50,7 %** sistema · **64,5 %** repeat · **8,6 %**
      ALS frente a **6,5 %** popularidad · **7 + 3** · **unos 50 %** de cada
      canasta es recompra.
