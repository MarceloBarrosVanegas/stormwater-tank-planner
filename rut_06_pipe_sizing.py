import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from operator import itemgetter
from functools import reduce
from line_profiler import profile

# append relative paths
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gui'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util', 'geometry'))

# variable global de rugosidad
from rut_00_app_config import geometry_vg
from rut_04_roughness import RoughnessGetter


class SeccionLlena:
    def __init__(self):
        pass


    @staticmethod
    def section_str2float(arr, return_b=False, return_all=False, sep="x"):

        r"""
        Converts an array of section dimension strings into a numeric array.

        The function parses strings that define geometric sections (circular, rectangular, trapezoidal or triangular)
         and returns specific dimensions as a float array.

        Dimension String Format:
        - Circular: "diameter" (e.g., "0.5")
        - Rectangular: "bottom base x height" (e.g., "2.0x1.5")
        - Trapezoidal: " top base x height x slope left x slope right" (e.g., "2.0x1.5x0.5x0.5")
        -Triangular: " 0 x height x slope left x slope right" (e.g., "0x1.5x0.25x0.25")

        Graphical representation:
                                                                                       ---
              ******            **************    ******************     ***********    │
           ************         *            *     ****************       *********     │
          **************        *            *      **************         *******      │  geom1
          **************        *            *       ************           *****       │
           ************         **************        **********             ***        │
              ******                                                          *         │
                                                                                       ---
          |----geom0---|        |----geom0---|        |-geom0-|            geom0 = 0

             Círculo              Rectángulo           Trapecio           Triángulo

        geom0: (circular:diámetro, rectangular:base, trapezoidal:base inferior, triangular: 0m)
        geom1: altura
        geom2: pendiente izquierda (vertical/horizontal)
        geom3: pendiente derecha (vertical/horizontal)

        """

        # crear series de pandas
        s = pd.Series(arr)

        # separar dimensiones de secciones
        dimensiones_df = s.str.split(sep, expand=True)

        dimensiones_array = dimensiones_df.to_numpy(dtype=float)
        l, dims = dimensiones_array.shape

        # Initialize geometry arrays
        geom = np.full((l, 4), np.nan)

        if dims == 1:
            geom[:, 0] = dimensiones_array[:, 0] #diametro

        elif dims == 2:
            geom[:, 0] = dimensiones_array[:, 0] # base
            geom[:, 1] = dimensiones_array[:, 1] # altura

        elif dims == 4:
            geom[:, 0] = dimensiones_array[:, 0]  # base
            geom[:, 1] = dimensiones_array[:, 1]  # altura
            geom[:, 2] = dimensiones_array[:, 2]  # talud izquierdo
            geom[:, 3] = dimensiones_array[:, 3]  # talud derecho

        else:
            pass

        if return_all:
            return geom

        #return specific dimension
        geom0 = geom[:, 0]
        geom1 = geom[:, 1]
        geom2 = geom[:, 2]
        geom3 = geom[:, 3]

        # get circular and prismatic indexes
        circular_index = (geom0 > 0) & np.isnan(geom1) & np.isnan(geom2) & np.isnan(geom3)
        prismatic_index = ~circular_index

        # zeros array
        out = np.zeros(shape=l, dtype=float)

        # Circular: return diameter
        out[circular_index] = geom0[circular_index]

        if return_b:
            base_superior_triangulo = (geom1 / geom2) + (geom1 / geom3)

            # return height for all prismatic sections
            out[prismatic_index] = np.where(geom0[prismatic_index] > 0, geom0[prismatic_index], base_superior_triangulo)
        else:
            # return height for all prismatic sections
            out[prismatic_index] = geom1[prismatic_index]

        return out

    #-------------------------------------------------------------------------------------------------
    def sll(self, D_int, S, Rug, Seccion):
        """
        Calculates full-flow capacity (Q) and velocity (V) using Manning's equation.

        This method computes the hydraulic properties for conduits flowing full, based on their
        geometry, slope, and roughness. It supports various cross-section shapes including
        circular, rectangular (open and closed), trapezoidal, and triangular.

        Args:
            D_int (np.ndarray): Array of strings with the internal dimensions of each section.
                                e.g., "0.5" for circular, "2.0x1.5" for rectangular.
            S (np.ndarray): Array of hydraulic slopes [m/m].
            Rug (np.ndarray): Array of Manning's roughness coefficients 'n'.
            Seccion (np.ndarray): Array of strings specifying the cross-section type for each conduit.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - q_sll (np.ndarray): Calculated full-flow capacity for each section [L/s].
                - v_sll (np.ndarray): Calculated full-flow velocity for each section [m/s].
        """

        q_sll = np.zeros(D_int.shape[0])
        v_sll = np.zeros(D_int.shape[0])
        secciones = np.unique(Seccion)

        # Iterate over each unique cross-section type present in the input
        for seccion in secciones:
            seccion_index = np.char.equal(Seccion, seccion)
            manning = Rug[seccion_index]
            pendiente = S[seccion_index]
            dimensiones_seccion = self.section_str2float(D_int[seccion_index], return_all=True)

            if seccion in ["circular"]:
                diametro_circular = dimensiones_seccion[:, 0] #diametro
                area = np.pi * (diametro_circular ** 2) / 4.0
                perimetro_mojado = np.pi * diametro_circular

            elif seccion in ["rectangular"]:
                base, altura = dimensiones_seccion[:, 0], dimensiones_seccion[:, 1]
                area = base * altura
                perimetro_mojado = 2 * (base + altura)

            elif seccion in ["rectangular_abierta"]:
                base, altura = dimensiones_seccion[:, 0], dimensiones_seccion[:, 1]
                area = base * altura
                perimetro_mojado = base + 2 * altura

            elif seccion in ["trapezoidal"]:
                base = dimensiones_seccion[:, 0]  # base inferior
                altura = dimensiones_seccion[:, 1]  # altura
                talud_izquierdo = dimensiones_seccion[:, 2]  # talud izq (vertical/horizontal)
                talud_derecho = dimensiones_seccion[:, 3]  # talud izq (vertical/horizontal)

                # Área: A = b*h + 0.5*m_i*h^2 + 0.5*m_d*h^2
                area = (base * altura) + 0.5 * talud_izquierdo * altura ** 2 + 0.5 * talud_derecho * altura ** 2
                # Perímetro mojado: P = b + h*sqrt(m_i^2 + 1) + h*sqrt(m_d^2 + 1)
                perimetro_mojado = base + altura * np.sqrt(talud_izquierdo ** 2 + 1) + altura * np.sqrt(talud_derecho ** 2 + 1)

            elif seccion in ["triangular"]:
                altura = dimensiones_seccion[:, 1] # altura
                talud_izquierdo = dimensiones_seccion[:, 2]  # talud izq (vertical/horizontal)
                talud_derecho = dimensiones_seccion[:, 3]  # talud izq (vertical/horizontal)

                # Área: A = 0.5*(m_i + m_d)*h^2
                area = 0.5 * (talud_izquierdo + talud_derecho) * altura ** 2
                # Perímetro mojado: P = h*sqrt(m_i^2 + 1) + h*sqrt(m_d^2 + 1)
                perimetro_mojado = altura * np.sqrt(talud_izquierdo ** 2 + 1) + altura * np.sqrt(talud_derecho ** 2 + 1)

            else:
                area = np.full(manning.shape, fill_value=np.nan)
                perimetro_mojado = np.full(manning.shape, fill_value=np.nan)


            # Calcular radio hidráulico evitando división por cero
            radio_hidraulico = area / perimetro_mojado

            # Ecuación de Manning para velocidad
            v_sll[seccion_index] = (1 / manning) * (radio_hidraulico ** (2.0 / 3.0)) * (np.abs(pendiente) ** (1.0 / 2.0))

            # Caudal = velocidad × área (multiplicado por 1000 para obtener L/s)
            q_sll[seccion_index] = v_sll[seccion_index] * area * 1000.0

        return q_sll, v_sll

    def get_SLL(self, m_ramales,ramal_value=None):
        # check for specific ramal
        if ramal_value:
            lista_ramales = [ramal_value]
        else:
            lista_ramales = m_ramales.keys()

        for i in lista_ramales:
            m_ramales[i]["Q_SLL"], m_ramales[i]["V_SLL"] = np.round(
                self.sll(
                    m_ramales[i]["D_int"],
                    m_ramales[i]["S"],
                    m_ramales[i]["Rug"],
                    m_ramales[i]["Seccion"],
                ),
                2,
            )

        return m_ramales


class SeccionParcialmenteLlena:
    def __init__(self):
        pass

    @staticmethod
    def section_str2float(arr, return_b=False, return_all=False, sep="x"):

        r"""
        Converts an array of section dimension strings into a numeric array.

        The function parses strings that define geometric sections (circular, rectangular, trapezoidal or triangular)
         and returns specific dimensions as a float array.

        Dimension String Format:
        - Circular: "diameter" (e.g., "0.5")
        - Rectangular: "bottom base x height" (e.g., "2.0x1.5")
        - Trapezoidal: " top base x height x slope left x slope right" (e.g., "2.0x1.5x0.5x0.5")
        -Triangular: " 0 x height x slope left x slope right" (e.g., "0x1.5x0.25x0.25")

        Graphical representation:
                                                                                       ---
              ******            **************    ******************     ***********    │
           ************         *            *     ****************       *********     │
          **************        *            *      **************         *******      │  geom1
          **************        *            *       ************           *****       │
           ************         **************        **********             ***        │
              ******                                                          *         │
                                                                                       ---
          |----geom0---|        |----geom0---|        |-geom0-|            geom0 = 0

             Círculo              Rectángulo           Trapecio           Triángulo

        geom0: (circular:diámetro, rectangular:base, trapezoidal:base inferior, triangular: 0m)
        geom1: altura
        geom2: pendiente izquierda (vertical/horizontal)
        geom3: pendiente derecha (vertical/horizontal)

        """

        # crear series de pandas
        s = pd.Series(arr)

        # separar dimensiones de secciones
        dimensiones_df = s.str.split(sep, expand=True)

        dimensiones_array = dimensiones_df.to_numpy(dtype=float)
        l, dims = dimensiones_array.shape

        # Initialize geometry arrays
        geom = np.full((l, 4), np.nan)

        if dims == 1:
            geom[:, 0] = dimensiones_array[:, 0] #diametro

        elif dims == 2:
            geom[:, 0] = dimensiones_array[:, 0] # base
            geom[:, 1] = dimensiones_array[:, 1] # altura

        elif dims == 4:
            geom[:, 0] = dimensiones_array[:, 0]  # base
            geom[:, 1] = dimensiones_array[:, 1]  # altura
            geom[:, 2] = dimensiones_array[:, 2]  # talud izquierdo
            geom[:, 3] = dimensiones_array[:, 3]  # talud derecho

        else:
            pass

        if return_all:
            return geom

        #return specific dimension
        geom0 = geom[:, 0]
        geom1 = geom[:, 1]
        geom2 = geom[:, 2]
        geom3 = geom[:, 3]

        # get circular and prismatic indexes
        circular_index = (geom0 > 0) & np.isnan(geom1) & np.isnan(geom2) & np.isnan(geom3)
        prismatic_index = ~circular_index

        # zeros array
        out = np.zeros(shape=l, dtype=float)

        # Circular: return diameter
        out[circular_index] = geom0[circular_index]

        if return_b:
            base_superior_triangulo = (geom1 / geom2) + (geom1 / geom3)

            # return height for all prismatic sections
            out[prismatic_index] = np.where(geom0[prismatic_index] > 0, geom0[prismatic_index], base_superior_triangulo)
        else:
            # return height for all prismatic sections
            out[prismatic_index] = geom1[prismatic_index]

        return out

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def circular_spll_root(ang_v, q_accu, q_sll):
        """
        Residuo adimensional para sección circular parcialmente llena (SPLL),
        optimizado para velocidad y robustez en solvers.

        Definición:
          φ(θ) = (θ/360 − sinθ / (2π)) · [ 1 − (360·sinθ)/(2π·θ) ]^(2/3)
          residual = φ(θ) − (q_accu / q_sll_array)

        """
        # Convert angle from degrees to radians for trigonometric functions
        ang_rad = np.radians(ang_v)
        sin_of_angle = np.sin(ang_rad)
        two_pi = 2.0 * np.pi

        # This term represents the ratio of the wetted area to the total pipe area (A/A_full)
        area_ratio = (ang_v / 360.0) - (sin_of_angle / two_pi)

        # This term is related to the ratio of the hydraulic radius to the full pipe hydraulic radius (R/R_full)
        hydraulic_radius_ratio_term = 1.0 - (360.0 * sin_of_angle) / (two_pi * ang_v)

        # The dimensionless flow relationship φ(θ) = (A/A_full) * (R/R_full)^(2/3)
        phi = area_ratio * (hydraulic_radius_ratio_term ** (2.0 / 3.0))

        # The residual is the difference between the calculated and target flow ratios
        residual = phi - (q_accu / q_sll)

        return residual

    @staticmethod
    def triangular_spll_root(calado, pendiente, rugosidad, q_target, talud_izquierdo, talud_derecho):
        """
        Calcula el residuo (Q_modelado - Q_objetivo) en m³/s para una sección triangular
        con taludes asimétricos, usando Manning.

        Parámetros (todos np.ndarray o broadcasteables)
        ----------
        calado : np.ndarray [m]
            Tirante (y)
        pendiente : np.ndarray [-]
            Pendiente S [m/m]
        rugosidad : np.ndarray [-]
            Coeficiente de Manning (n)
        q_target : np.ndarray [L/s]
            Caudal objetivo
        talud_izquierdo : float o np.ndarray [-]
            Talud izquierdo (m_i:1)
        talud_derecho : float o np.ndarray [-]
            Talud derecho (m_d:1)

        Retorna
        -------
        residual : np.ndarray [m³/s]
            Q_modelado - Q_objetivo. NaN si no es físico.
        """

        # Área total: suma de dos triángulos
        area_total = 0.5 * (talud_izquierdo + talud_derecho) * calado ** 2  # [m²]

        # Perímetro: dos hipotenusas
        L_izq = calado * np.sqrt(1 + talud_izquierdo ** 2)
        L_der = calado * np.sqrt(1 + talud_derecho ** 2)
        perimetro_total = L_izq + L_der  # [m]

        # Radio hidráulico
        radio_hidraulico = area_total / perimetro_total

        # Caudal modelado con Manning
        caudal_modelado = (
                (1.0 / rugosidad) *
                area_total *
                (radio_hidraulico ** (2.0 / 3.0)) *
                np.sqrt(pendiente)
        )

        # Residual: Q_modelado - Q_objetivo [m³/s]
        residual = caudal_modelado - (q_target / 1000.0)

        return residual

    @staticmethod
    def rectangular_spll_root( calado, base, pendiente, rugosidad, q_target):
        r"""
        Dimensiona la seccion parcialmente llena (SPLL), de un canal rectangular con Manning y devuelve el
        residuo de caudal (Q_model - Q_target) en m³/s.

        Modelo:
            Q_model = (1/n) * A * R^(2/3) * S^(1/2)
          con:
            A = b * y                 (área mojada)        [m²]
            P = b + 2*y               (perímetro mojado)   [m]
            R = A / P                 (radio hidráulico)   [m]
            Q_target = q_accu / 1000  (convierte L/s → m³/s)

        Parámetros
        ----------
        calado : float | array_like, [m]
            Tirante (y).
        base : float | array_like, [m]
            Ancho de fondo (b).
        pendiente : float | array_like, [-]
            Pendiente hidráulica (adimensional).
        rugosidad : float | array_like, [-]
            Coeficiente de Manning (n).
        q_target : float | array_like, [L/s]
            Caudal objetivo.

        Retorna
        -------
        residual : np.ndarray, [m³/s]
            Q_model - Q_target para cada sección (NaN donde los insumos no son físicos).
        """


        # Geometría de la sección rectangular
        A = base * calado                 # área mojada [m²]
        P = base + 2.0 * calado           # perímetro mojado [m]
        R =  A / P                        # radio hidráulico [m]

        # Caudal modelado con Manning únicamente en posiciones válidas
        q_model = (A * (R ** (2.0 / 3.0)) * np.sqrt(pendiente)) / rugosidad

        # Residuo (modelo - objetivo)
        residual = q_model - (q_target / 1000.0)

        return residual

    @staticmethod
    def trapezoidal_spll_root(calado, base, pendiente, rugosidad, q_target, talud_izquierdo, talud_derecho):
        """
        Calcula el residuo del caudal (Q_modelado - Q_objetivo) en m³/s para una sección
        trapezoidal parcialmente llena usando la fórmula de Manning.

        Los taludes se definen como relaciones horizontales por unidad de tirante vertical (m:1).
        Por ejemplo:
            talud_izquierdo = 2.0 → por cada 1 m de profundidad, el talud avanza 2 m hacia la izquierda.

        Descomposición geométrica:
            - Área total = área del triángulo izquierdo + área del rectángulo central + área del triángulo derecho
            - Perímetro mojado = base + longitud del talud izquierdo + longitud del talud derecho
            - Radio hidráulico = área / perímetro

        Parámetros
        ----------
        calado : np.ndarray
            Tirante de agua (y) [m].
        base : np.ndarray
            Ancho de fondo del canal (b) [m].
        pendiente : np.ndarray
            Pendiente longitudinal del fondo del canal (S) [adimensional, m/m].
        rugosidad : np.ndarray
            Coeficiente de Manning (n) [adimensional].
        q_target : np.ndarray
            Caudal objetivo para la sección (Q) [L/s].
        talud_izquierdo : np.ndarray
            Talud del lado izquierdo, expresado como proyección horizontal por unidad de tirante (m_i:1) [adimensional].
        talud_derecho : np.ndarray
            Talud del lado derecho, expresado como proyección horizontal por unidad de tirante (m_d:1) [adimensional].

        Retorna
        -------
        residual : np.ndarray
            Diferencia entre el caudal calculado y el objetivo, en m³/s.
            Devuelve NaN en posiciones donde los valores no sean físicamente válidos.
        """

        # Área de la sección
        area_izquierda = 0.5 * talud_izquierdo * calado ** 2
        area_central = base * calado
        area_derecha = 0.5 * talud_derecho * calado ** 2
        area_total = area_izquierda + area_central + area_derecha  # [m²]

        # Perímetro mojado
        longitud_talud_izquierdo = calado * np.sqrt(1 + talud_izquierdo ** 2)
        longitud_talud_derecho = calado * np.sqrt(1 + talud_derecho ** 2)
        perimetro_total = base + longitud_talud_izquierdo + longitud_talud_derecho  # [m]

        # Radio hidráulico
        radio_hidraulico = np.where(perimetro_total > 0, area_total / perimetro_total, 0.0)

        # Caudal modelado con Manning: Q = (1/n) * A * R^(2/3) * sqrt(S)
        caudal_modelado = (
                (1.0 / rugosidad) *
                area_total *
                (radio_hidraulico ** (2.0 / 3.0)) *
                np.sqrt(pendiente)
        )

        # Residual: Q_modelado - Q_objetivo [m³/s]
        residual = caudal_modelado - (q_target / 1000.0)

        return residual

    #-----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def seccion_init_circular(q, manning, pendiente):
        """
        diámetro de una tubería circular parcialmente llena utilizando Manning

        Fórmula base:
        q = (D^(8/3)) / (7257.15 * manning * (2πθ)^(2/3)) * sqrt(pendiente) * (2πθ - 360·sinθ)^(5/3)

        Donde:
        - θ° = 2·arccos(1 - 2h/D) [ángulo central en grados]
        - θ = θ° * π/180 [conversión a radianes para cálculos]

        Args:
            q (float array): Caudal volumétrico en Litros por segundo (L/s)
            manning (float array): Coeficiente de rugosidad de Manning
            pendiente (float array): Pendiente longitudinal de la tubería (m/m)

        Returns:
            float array: Diámetro interno requerido en metros
        """

        # Relación altura sobre diámetro
        h_D = geometry_vg.map_calado_dict.get('sub-critico', 0.82)  # Valor por defecto si está vacío

        # Convertir caudal de L/s a m³/s
        q_m3s = q / 1000.0

        # Ángulo central en grados: θ° = 2·arccos(1 - 2h/D)
        theta_grados = 2 * np.degrees(np.arccos(1 - 2 * h_D))

        # Convertir a radianes para funciones trigonométricas
        theta_rad = np.radians(theta_grados)

        try:
            # Factores de la ecuación
            # Denominador: 7257.15 * manning * (2πθ°)^(2/3)
            factor_denominador = 7257.15 * manning * (2 * np.pi * theta_grados) ** (2.0 / 3.0)

            # Área efectiva: (2πθ° - 360·sin(θ))^(5/3)
            area_efectiva = (2 * np.pi * theta_grados - 360.0 * np.sin(theta_rad)) ** (5.0 / 3.0)

            # Término derecho de la ecuación despejada para D
            termino_derecho = (q_m3s * factor_denominador) / (area_efectiva * np.sqrt(pendiente))

            # Diámetro: D = (término_derecho)^(3/8)
            diametro = termino_derecho ** (3.0 / 8.0)

        except (ValueError, ZeroDivisionError, RuntimeWarning):
            # Retornar NaN para valores inválidos
            return np.full_like(q, np.nan, dtype=float)

        return diametro

    def seccion_init_rectangular(self, i, q_target, rugosidad, pendiente):
        """
        Estima dimensiones iniciales (b, y) para una sección rectangular parcialmente llena
        que cumpla un caudal objetivo q_target, resolviendo el tirante con rugosidad mediante una raíz.

        Supuestos y convenciones
        ------------------------
        - Régimen uniforme con la fórmula de rugosidad.
        - Semilla geométrica: sección rectangular eficiente (b = 2*y).
        - Unidades: q_target [L/s], rugosidad = n [-], pendiente = S [-] (m/m).
        - y se ajusta por un factor h/D proveniente de la configuración global (véase 'h_D').

        Pasos del procedimiento
        -----------------------
        1) Calcula un ancho inicial b_min a partir de la sección eficiente:
               y_eff = [ ( (q_target/1000) * n ) / ( 2^(1/3) * sqrt(S) ) ]^(3/8)
               b_min = 2*y_eff
           (q_target/1000 convierte L/s → m³/s)
        2) Aplica una dimensión mínima para la sección rectangular y redondea b a 0.1 m.
        3) Resuelve el tirante y con un método de raíces sobre el residual
           `rectangular_spll_root(y, b, S, n, q_target)`:
             - Intento principal: `scipy.optimize.brentq` en [0.001, 5*b]
             - Respaldo: `scipy.optimize.root(..., method="hybr")`
        4) Ajusta el tirante por el factor h_D (altura/diámetro)
           y vuelve a aplicar la dimensión mínima.
        5) Devuelve (b, y) en metros.

        Parámetros
        ----------
        q_target : array_like de float
            Caudal objetivo por sección [L/s].
        rugosidad : array_like de float
            Coeficiente de rugosidad, n [-].
        pendiente : array_like de float
            Pendiente hidráulica S [-] (m/m).

        Retorna
        -------
        b : np.ndarray
            Ancho de fondo estimado por sección [m].
        y : np.ndarray
            Tirante estimado por sección [m] (ajustado por h_D y con dimensión mínima).

        Notas
        -----
        - `h_D` proviene de `geometry_vg.map_calado_dict` (por defecto 0.75). Si no requieres
          normalizar por h/D en rectangulares, puedes omitir ese ajuste en tu flujo.
        - Este procedimiento usa `rectangular_spll_root` para evaluar el residuo de caudal.
        """

        # Relación calado sobre dimension vertical
        h_D = geometry_vg.map_calado_dict.get('sub-critico', 0.82)  # Valor por defecto si está vacío

        #sección rectangular eficiente (b = (1.5 a 2) *y)
        b_min = 2.0 * (((q_target / 1000.0) * rugosidad) / (2 ** (1 / 3) * (pendiente ** 0.5))) ** (3 / 8)

        # asegurar la dimensión mínima y redondeo de b a 0.1 m
        dimension_minima_rectangular = geometry_vg.parametros_basicos['rectangular']['dimension_minima']
        b_min = np.where(b_min >= dimension_minima_rectangular, b_min, dimension_minima_rectangular)
        b = np.around(b_min, decimals=1)
        b = np.maximum.accumulate(b)

        # Resolver y para cada sección usando el residual rectangular_spll_root
        all_args = [b, pendiente, rugosidad, q_target]
        root_func = self.rectangular_spll_root

        # Find water depth for prismatic section
        calado = self._find_roots(
            root_func=root_func,
            ramal_id=i,
            all_args=all_args,
            lower_bound=0.001,
            upper_bound=b.max() * 2,
        )
        y = np.round(calado, 3)


        # ajuste por h/D y asegurar dimensión mínima
        y =  y / h_D
        y = np.where(y >= dimension_minima_rectangular, y, dimension_minima_rectangular)
        y = np.around(y, decimals=1)
        y = np.maximum.accumulate(y)

        return b, y

    def seccion_init_trapezoidal(self, i, q_target, rugosidad, pendiente, talud_izquierdo=None, talud_derecho=None):
        """
        Estima dimensiones iniciales (b, y) para una sección trapezoidal parcialmente llena,
        usando la fórmula de Manning.

        Supuestos
        ---------
        - Régimen uniforme.
        - Por defecto: sección trapezoidal eficiente simétrica con taludes 1:√3 (≈1.732:1),
          que minimiza el perímetro mojado (más eficiente hidráulicamente).
        - Si se especifican taludes distintos, se usa una estimación robusta basada en
          sección rectangular eficiente.
        - Unidades: q_target [L/s], rugosidad [-], pendiente [m/m].

        Retorna
        -------
        b : np.ndarray [m]
            Ancho de fondo estimado.
        y : np.ndarray [m]
            Tirante estimado (ajustado por h_D y dimensión mínima).
        talud_izquierdo: np.ndarray [-]
            Talud izquierdo usado (horizontal:vertical), devuelto para trazabilidad.
        talud_derecho: np.ndarray [-]
            Talud derecho usado (horizontal:vertical), devuelto para trazabilidad.
        """

        # Factor de calado
        h_D = geometry_vg.map_calado_dict.get('sub-critico', 0.82)

        # Valor por defecto: canal eficiente (60°)
        talud_defecto = geometry_vg.parametros_basicos['trapezoidal']['talud_optimo']# ≈1.732

        # Asegurar forma común
        shape = q_target.shape

        # Asignar valores por defecto y convertir a arrays con forma correcta
        if talud_izquierdo is None:
            talud_izquierdo = np.full(shape, talud_defecto)
        if talud_derecho is None:
            talud_derecho = np.full(shape, talud_defecto)

        # Dimensión mínima
        dim_min = geometry_vg.parametros_basicos['trapezoidal']['dimension_minima']

        # ¿Es el caso eficiente? (taludes ≈ √3 y simétricos)
        es_eficiente = np.allclose(talud_izquierdo, talud_defecto) & np.allclose(talud_derecho, talud_defecto)

        q_m3s = q_target / 1000.0  # L/s → m³/s

        if es_eficiente:
            # Fórmula analítica para trapecio eficiente (60°)
            y_eff = (q_m3s * rugosidad * (2 ** (2 / 3)) / (talud_defecto * np.sqrt(pendiente))) ** (3 / 8)
            b_min = (2 / talud_defecto) * y_eff
        else:
            # Estimación robusta: como rectangular eficiente
            y_eff = (q_m3s * rugosidad / (2 ** (1 / 3) * np.sqrt(pendiente))) ** (3 / 8)
            b_min = geometry_vg.parametros_basicos['rectangular']['relacion_b_y_optima'] * y_eff

        # Ajustar dimensión mínima y redondear b
        b_min = np.where(b_min >= dim_min, b_min, dim_min)
        b = np.round(b_min, decimals=1)
        b = np.maximum.accumulate(b)

        # Resolver y para cada sección usando el residual rectangular_spll_root
        all_args = [b, pendiente, rugosidad, q_target, talud_izquierdo, talud_derecho]
        root_func = self.trapezoidal_spll_root

        # Find water depth for prismatic section
        calado = self._find_roots(
            root_func=root_func,
            ramal_id=i,
            all_args=all_args,
            lower_bound=0.001,
            upper_bound=b.max() * 2,
        )
        y = np.round(calado, 3)

        # Ajuste final por h/D y dimensión mínima
        y = y / h_D
        y = np.where(y >= dim_min, y, dim_min)
        y = np.round(y, decimals=1)
        y =  np.maximum.accumulate(y)

        return b, y, talud_izquierdo, talud_derecho

    def seccion_init_triangular(self, i, q_target, rugosidad, pendiente, talud_izquierdo=None, talud_derecho=None):
        """
        Estima el tirante inicial (y) para una sección triangular parcialmente llena,
        usando la fórmula de Manning.

        Supuestos
        ---------
        - Régimen uniforme.
        - Por defecto: sección triangular con taludes 1:1 (45°), caso común y simétrico.
        - Si se especifican taludes distintos, se usa una estimación robusta.
        - Unidades: q_target [L/s], rugosidad [-], pendiente [m/m].

        Retorna
        -------
        y : np.ndarray [m]
            Tirante estimado (ajustado por h_D y dimensión mínima).
        talud_izquierdo : np.ndarray [-]
            Talud izquierdo usado (horizontal:vertical), devuelto para trazabilidad.
        talud_derecho : np.ndarray [-]
            Talud derecho usado (horizontal:vertical), devuelto para trazabilidad.
        """

        # Factor de calado
        h_D = geometry_vg.map_calado_dict.get('sub-critico', 0.82)

        # Valor por defecto: taludes 1:1 (45°), no √3
        talud_defecto = geometry_vg.parametros_basicos['triangular']['talud_optimo']# ≈ 1

        # Asegurar forma común
        shape = q_target.shape

        # Asignar valores por defecto y convertir a arrays con forma correcta
        if talud_izquierdo is None:
            talud_izquierdo = np.full(shape, talud_defecto)
        if talud_derecho is None:
            talud_derecho = np.full(shape, talud_defecto)

        # Dimensión mínima
        dim_min = geometry_vg.parametros_basicos.get('triangular', {}).get(
            'dimension_minima',
            geometry_vg.parametros_basicos['rectangular']['dimension_minima']
        )

        # Convertir caudal a m³/s
        q_m3s = q_target / 1000.0

        # --- Estimación inicial del tirante ---
        # Usamos una estimación robusta basada en sección rectangular eficiente
        # (b = 2y), como punto de partida genérico
        y_eff = (q_m3s * rugosidad / (2 ** (1 / 3) * np.sqrt(pendiente))) ** (3 / 8)

        # Ajustar dimensión mínima y redondear
        y_min = np.where(y_eff >= dim_min, y_eff, dim_min)
        y_init = np.round(y_min, decimals=1)
        y_init = np.maximum.accumulate(y_init)

        all_args = [pendiente, rugosidad, q_target, talud_izquierdo, talud_derecho]
        root_func = self.triangular_spll_root

        # Find water depth for prismatic section
        calado = self._find_roots(
            root_func=root_func,
            ramal_id=i,
            all_args=all_args,
            lower_bound=0.001,
            upper_bound=y_init.max() * 2,
        )
        y = np.round(calado, 3)

        # Ajuste final por h/D y dimensión mínima
        y = y / h_D
        y = np.where(y >= dim_min, y, dim_min)
        y = np.round(y, decimals=1)
        y = np.maximum.accumulate(y)

        return y, talud_izquierdo, talud_derecho

    #-----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_circular_velocity_ratio(ang_v):
        """
        Calcula la relación de velocidad (v/V) para una sección circular parcialmente llena.

        La relación se deriva de la ecuación de Manning, donde v/V = (R/R_full)^(2/3).

        Args:
            ang_v (np.ndarray): Ángulo central de llenado en grados (°).

        Returns:
            np.ndarray: Relación adimensional de velocidad (v/V).
        """
        # Evita la división por cero cuando el ángulo es 0
        with np.errstate(divide='ignore', invalid='ignore'):
            # Término del radio hidráulico: R/R_full = 1 - (360 * sin(θ)) / (2π * θ)
            # donde θ está en grados para el término 'ang_v' y en radianes para sin()
            hydraulic_radius_ratio = 1.0 - (360.0 * np.sin(np.radians(ang_v))) / (2 * np.pi * ang_v)

            # Relación de velocidad: (R/R_full)^(2/3)
            velocity_ratio = np.power(hydraulic_radius_ratio, 2.0 / 3.0)

        # Reemplaza NaN (resultado de 0/0) con 0 para tuberías vacías
        return np.nan_to_num(velocity_ratio)

    @staticmethod
    def get_circular_depth_ratio(ang_v):
        """
        Calcula la relación de calado (h/D) para una sección circular parcialmente llena.

        La relación se basa en la geometría del segmento circular.
        h/D = 0.5 * (1 - cos(θ/2))

        Args:
            ang_v (np.ndarray): Ángulo central de llenado en grados (°).

        Returns:
            np.ndarray: Relación adimensional de calado (h/D).
        """
        # Ángulo en radianes para la función coseno
        angle_rad_half = np.radians(ang_v / 2.0)

        # Fórmula geométrica para la relación de calado
        depth_ratio = 0.5 * (1.0 - np.cos(angle_rad_half))

        return depth_ratio

    @staticmethod
    def get_prismatic_velocity(radio_hidraulico, pendiente, rugosidad):
        """
        Calculates velocity for a triangular section using Manning's equation.

        Args:
            radio_hidraulico (np.ndarray): Hydraulic radius (R) [m].
            pendiente (np.ndarray): Channel slope (S) [m/m].
            rugosidad (np.ndarray): Manning's roughness coefficient (n).

        Returns:
            np.ndarray: Flow velocity (v) [m/s].
        """
        return (radio_hidraulico ** (2.0 / 3.0)) * np.sqrt(pendiente) / rugosidad

    @staticmethod
    def get_prismatic_hydraulic_radius(seccion, calado=None, base=None, talud_izq=None, talud_der=None):
        """
        Calculates hydraulic radius based on section type.

        For prismatic (rectangular, triangular, trapezoidal): Uses depth (calado) to compute wetted area and perimeter.

        Args:
            seccion (str): Section type (e.g., "circular", "rectangular").
            calado (float or np.ndarray): Water depth [m] (for prismatic sections).
            base (float or np.ndarray): Bottom width [m] (for rectangular/trapezoidal).
            talud_izq (float or np.ndarray): Left slope (for triangular/trapezoidal).
            talud_der (float or np.ndarray): Right slope (for triangular/trapezoidal).

        Returns:
            np.ndarray: Hydraulic radius [m].
        """

        if seccion in ["rectangular", "rectangular_abierta"]:
            if calado is None or base is None:
                raise ValueError("For rectangular sections, provide 'calado' and 'base'.")
            area = base * calado
            perimetro = base + 2 * calado
            return area / perimetro

        elif seccion == "triangular":
            if calado is None or talud_izq is None or talud_der is None:
                raise ValueError("For triangular sections, provide 'calado', 'talud_izq', 'talud_der'.")
            area = 0.5 * (talud_izq + talud_der) * calado ** 2
            perimetro = calado * (np.sqrt(talud_izq ** 2 + 1) + np.sqrt(talud_der ** 2 + 1))
            return area / perimetro

        elif seccion == "trapezoidal":
            if calado is None or base is None or talud_izq is None or talud_der is None:
                raise ValueError("For trapezoidal sections, provide 'calado', 'base', 'talud_izq', 'talud_der'.")
            area = base * calado + 0.5 * (talud_izq + talud_der) * calado ** 2
            perimetro = base + calado * (np.sqrt(talud_izq ** 2 + 1) + np.sqrt(talud_der ** 2 + 1))
            return area / perimetro

        else:
            raise ValueError(f"Unsupported section type: {seccion}")

    @staticmethod
    def _find_roots(root_func, ramal_id, all_args, lower_bound, upper_bound):
        """
        Iteratively finds the root for a given hydraulic function for multiple pipe sections.

        This method serves as a generic solver that iterates over a set of section indices,
        calculating a key hydraulic variable for each. It primarily uses the efficient `brentq`
        method for root-finding within a defined interval and falls back to the more robust
        `root` method if `brentq` fails.

        The variable being solved for depends on the `root_func` provided:
        - For prismatic sections (rectangular, triangular, etc.), it solves for the water depth ('calado').
        - For circular sections, it solves for the central fill angle ('ang') in degrees.

        Args:
            root_func (callable): The function whose root is to be found (e.g., `self.circular_spll_root`).
            ramal_id (str or int): The identifier of the current branch, used for logging errors.
            all_args (list[np.ndarray]): A list of full NumPy arrays that will be passed as arguments to `root_func`.
            lower_bound (float): The lower boundary for the `brentq` search.
            upper_bound (float): The upper boundary for the `brentq` search.

        Returns:
            np.ndarray: An array containing the solved roots (depth or angle) for each section.
        """


        # Transpose the list of argument arrays into a list of argument tuples.
        # Example: [ [a1, a2], [b1, b2] ] -> [ (a1, b1), (a2, b2) ]
        args_per_section = list(zip(*all_args))

        # Attempt to find the root using brentq method
        try:
            roots = np.array([
                opt.brentq(
                    root_func,
                    lower_bound,
                    upper_bound,
                    xtol=1e-3,
                    args=args_tuple,
                ) for args_tuple in args_per_section
            ])

        except Exception as e:
            # NO FALLBACK - fail explicitly with detailed error message
            raise RuntimeError(
                f"Error in brentq for {root_func.__name__} in Ramal {ramal_id}.\n"
                f"  Lower bound: {lower_bound}, Upper bound: {upper_bound}\n"
                f"  Number of sections: {len(args_per_section)}\n"
                f"  First section args: {args_per_section[0] if args_per_section else 'N/A'}\n"
                f"  Original error: {e}"
            ) from e

        return roots

    # -----------------------------------------------------------------------------------------------------------------
    def get_SPLL(self, m_ramales, ramal_value=None):
        """
        Calculates hydraulic properties for partially full sections (SPLL) in each branch.

        For each branch (ramal), computes the fill angle, flow ratio, depth ratio, velocity ratio,
        water depth, and velocity for each section type (circular, rectangular, trapezoidal, triangular).
        Results are stored in the m_ramales dictionary.

        Args:
            m_ramales (dict): Dictionary containing hydraulic and geometric data for each branch.
            ramal_value (str, optional): If provided, only processes the specified branch.

        Returns:
            dict: Updated m_ramales with SPLL hydraulic results for each section.
        """

        if ramal_value:
            lista_ramales = [ramal_value]
        else:
            lista_ramales = list(m_ramales.keys())

        for i in lista_ramales:
            # Initialize empty arrays for hydraulic results
            empty_array = np.zeros(shape=len(m_ramales[i]["Pozo"]), dtype=geometry_vg.type_values_dict['S'])
            # crear arrays vacíos para llenar
            m_ramales[i]["ang"] = empty_array.copy()  # Fill angle (circular)
            m_ramales[i]["q/Q"] = empty_array.copy()  # Flow ratio
            m_ramales[i]["h/D"] = empty_array.copy()  # Depth ratio
            m_ramales[i]["v/V"] = empty_array.copy()  # Velocity ratio
            m_ramales[i]["h"] = empty_array.copy()  # Water depth
            m_ramales[i]["v"] = empty_array.copy()  # Velocity

            # Get unique section types in the branch
            secciones = np.unique(m_ramales[i]["Seccion"])

            # Replace zero flows to avoid division errors in root finding
            q = m_ramales[i]["q_accu"].copy()
            es_caudal_insignificante = m_ramales[i]["q_accu"] < 0.001
            es_caudal_significativo = np.invert(es_caudal_insignificante)
            q[es_caudal_insignificante] = 0.001

            # Get section dimensions as a float array
            dimensiones = self.section_str2float(m_ramales[i]["D_int"], return_all=True)

            for _seccion in secciones:
                # Indices for current section type
                seccion_index = np.trim_zeros(np.char.equal(m_ramales[i]["Seccion"], _seccion).nonzero()[0])

                # Mask for sections with significant flow
                is_flow_present = es_caudal_significativo[seccion_index]

                # Extract geometric parameters for current section
                base = np.nan_to_num(dimensiones[:, 0])[seccion_index]
                calado_max = np.nan_to_num(dimensiones[:, 1])[seccion_index]
                talud_izquierdo = dimensiones[:, 2][seccion_index]
                talud_derecho = dimensiones[:, 3][seccion_index]

                # Extract hydraulic parameters
                Q_SLL = m_ramales[i]["Q_SLL"][seccion_index]
                V_SLL = m_ramales[i]["V_SLL"][seccion_index]
                q_accu = m_ramales[i]["q_accu"][seccion_index]
                pendiente =  m_ramales[i]["S"][seccion_index]
                rugosidad = m_ramales[i]["Rug"][seccion_index]
                q_target = q[seccion_index]


                if _seccion in ["circular"]:
                    # Prepare arguments for root finding (angle)
                    all_args = [q_target, Q_SLL]

                    angulo = self._find_roots(
                            root_func =  self.circular_spll_root,
                            ramal_id=i,
                            all_args=all_args,
                            lower_bound=1,
                            upper_bound=10000,
                            )

                    # Limit angle to 360 degrees for partially full pipe
                    angulo[angulo >= 360] = 360

                    # Calculate depth and velocity ratios from angle
                    h_D_ratio = np.round(self.get_circular_depth_ratio(angulo), 3)
                    v_V_ratio = np.round(self.get_circular_velocity_ratio(angulo), 3)

                    # Calculate actual water depth and velocity
                    calado = np.round(h_D_ratio * base, 3)
                    velocidad = np.round(v_V_ratio * V_SLL, 3)

                    # Store angle, masking out zero-flow sections
                    m_ramales[i]["ang"][seccion_index] = angulo * is_flow_present

                else:
                    # Prepare arguments and root function for prismatic sections
                    if _seccion in ["rectangular", "rectangular_abierta"]:
                        all_args = [base, pendiente, rugosidad, q_target]
                        root_func = self.rectangular_spll_root

                    elif _seccion in ["triangular"]:
                        all_args = [pendiente, rugosidad, q_target, talud_izquierdo, talud_derecho]
                        root_func = self.triangular_spll_root

                    elif _seccion in ["trapezoidal"]:
                        all_args = [base, pendiente, rugosidad, q_target, talud_izquierdo, talud_derecho]
                        root_func = self.trapezoidal_spll_root

                    else:
                        raise ValueError(f"Unsupported section type: {_seccion}")


                    # Find water depth for prismatic section
                    calado = self._find_roots(
                        root_func=root_func,
                        ramal_id=i,
                        all_args=all_args,
                        lower_bound=0.001,
                        upper_bound=calado_max.max() * 2,
                    )
                    calado =  np.round(calado, 3)

                    # Calculate hydraulic radius and velocity
                    radio_hidraulico = self.get_prismatic_hydraulic_radius(
                        seccion=_seccion,
                        calado=calado,
                        base=base,
                        talud_izq=talud_izquierdo,
                        talud_der=talud_derecho
                    )
                    velocidad = np.round(self.get_prismatic_velocity(radio_hidraulico, pendiente, rugosidad), 3)


                # Calculate ratios and store results, masking out zero-flow sections
                q_Q_ratio = np.round(q_accu / Q_SLL, 4)
                h_D_ratio = np.round(calado / base if _seccion == 'circular' else calado / calado_max, 4)
                v_V_ratio = np.round(velocidad / V_SLL, 4)

                m_ramales[i]["q/Q"][seccion_index] = q_Q_ratio * is_flow_present
                m_ramales[i]["h/D"][seccion_index] = h_D_ratio * is_flow_present
                m_ramales[i]["v/V"][seccion_index] = v_V_ratio * is_flow_present
                m_ramales[i]["h"][seccion_index] = calado * is_flow_present
                m_ramales[i]["v"][seccion_index] = velocidad * is_flow_present


        return m_ramales


class PipeSizing:
    def __init__(self, m_ramales=None):
        self.m_ramales = m_ramales
        self.spll =  SeccionParcialmenteLlena()
        self.sll =  SeccionLlena()
        self.hydraulic_conditions = HydraulicConditions()
        self.roughness  = RoughnessGetter()
        self.redo_ramales_list = []

    @staticmethod
    def section_str2float(arr, return_b=False, return_all=False, sep="x"):

        r"""
        Converts an array of section dimension strings into a numeric array.

        The function parses strings that define geometric sections (circular, rectangular, trapezoidal or triangular)
         and returns specific dimensions as a float array.

        Dimension String Format:
        - Circular: "diameter" (e.g., "0.5")
        - Rectangular: "bottom base x height" (e.g., "2.0x1.5")
        - Trapezoidal: " top base x height x slope left x slope right" (e.g., "2.0x1.5x0.5x0.5")
        -Triangular: " 0 x height x slope left x slope right" (e.g., "0x1.5x0.25x0.25")

        Graphical representation:
                                                                                       ---
              ******            **************    ******************     ***********    │
           ************         *            *     ****************       *********     │
          **************        *            *      **************         *******      │  geom1
          **************        *            *       ************           *****       │
           ************         **************        **********             ***        │
              ******                                                          *         │
                                                                                       ---
          |----geom0---|        |----geom0---|        |-geom0-|            geom0 = 0

             Círculo              Rectángulo           Trapecio           Triángulo

        geom0: (circular:diámetro, rectangular:base, trapezoidal:base inferior, triangular: 0m)
        geom1: altura
        geom2: pendiente izquierda (vertical/horizontal)
        geom3: pendiente derecha (vertical/horizontal)

        """

        # crear series de pandas
        s = pd.Series(arr)

        # separar dimensiones de secciones
        dimensiones_df = s.str.split(sep, expand=True)

        dimensiones_array = dimensiones_df.to_numpy(dtype=float)
        l, dims = dimensiones_array.shape

        # Initialize geometry arrays
        geom = np.full((l, 4), np.nan)

        if dims == 1:
            geom[:, 0] = dimensiones_array[:, 0] #diametro

        elif dims == 2:
            geom[:, 0] = dimensiones_array[:, 0] # base
            geom[:, 1] = dimensiones_array[:, 1] # altura

        elif dims == 4:
            geom[:, 0] = dimensiones_array[:, 0]  # base
            geom[:, 1] = dimensiones_array[:, 1]  # altura
            geom[:, 2] = dimensiones_array[:, 2]  # talud izquierdo
            geom[:, 3] = dimensiones_array[:, 3]  # talud derecho

        else:
            pass

        if return_all:
            return geom

        #return specific dimension
        geom0 = geom[:, 0]
        geom1 = geom[:, 1]
        geom2 = geom[:, 2]
        geom3 = geom[:, 3]

        # get circular and prismatic indexes
        circular_index = (geom0 > 0) & np.isnan(geom1) & np.isnan(geom2) & np.isnan(geom3)
        prismatic_index = ~circular_index

        # zeros array
        out = np.zeros(shape=l, dtype=float)

        # Circular: return diameter
        out[circular_index] = geom0[circular_index]

        if return_b:
            base_superior_triangulo = (geom1 / geom2) + (geom1 / geom3)

            # return height for all prismatic sections
            out[prismatic_index] = np.where(geom0[prismatic_index] > 0, geom0[prismatic_index], base_superior_triangulo)
        else:
            # return height for all prismatic sections
            out[prismatic_index] = geom1[prismatic_index]

        return out

    def select_internal_diameter(self, m_ramales,i, index_seccion, seccion="circular"):

        # Extract the initial internal diameters for the given section indices and convert
        diameter_init = self.section_str2float(m_ramales[i]["D_int"][index_seccion])

        # Get a unique set of materials for the given section indices
        materiales = set(m_ramales[i]["Material"][index_seccion])
        secciones =  set(m_ramales[i]["Seccion"][index_seccion])

        count = 0
        # Initialize the condition for while loop
        cond = False
        while not cond and count < 10:
            count = count + 1

            if seccion == "circular":

                # Iterate over each unique material
                for j in materiales:
                    for seccion in secciones:
                        # Set the condition to True. It will be set to False later if a condition is not met.
                        cond = True

                        # Get the internal diameter for the current material 'j'
                        diametro_interno_minimo = self.get_internal_diameter_by_material(j)

                        # Find indices where the material type matches the current material 'j'
                        index_material = m_ramales[i]["Material"][index_seccion] == j

                        # Search for the positions in the sorted array where D_init_circular should be inserted
                        index_diametro = np.searchsorted(diametro_interno_minimo, diameter_init[index_material])

                        try:
                            # Try to assign the diameter values based on the searchsorted indices
                            diametro_interno = diametro_interno_minimo[index_diametro]
                        except Exception as e:

                            # determinar que posición de index_diametro esta fuera de rango
                            filtro_diametro_fuera_rango = index_diametro >= diametro_interno_minimo.size
                            filtro_diametro_fuera_rango = filtro_diametro_fuera_rango.nonzero()[0]

                            #revisar si la dimension vertical maxima de el array de diametro_interno_minimo es similar a la dimension vertical que causa el error en diametro_init
                            condicion_equiparar_diametros = np.isclose(diameter_init[filtro_diametro_fuera_rango], diametro_interno_minimo.max(), rtol=0.2)

                            #determinar el primer indice donde ya no se puede utilizar la misma dimension vertical, para luego pasarme a otro material
                            position_first_false = np.where(~condicion_equiparar_diametros)

                            lista_posiciones_false = position_first_false[0]
                            if lista_posiciones_false.size == 0:
                                position_first_false_index = diameter_init.size +1
                            else:
                                position_first_false_index = filtro_diametro_fuera_rango[position_first_false[0].min()]

                            for pos in filtro_diametro_fuera_rango:
                                if pos < position_first_false_index:
                                    #asignar la dimension vertical maxima
                                    diameter_init[pos] = diametro_interno_minimo.max()
                                else:

                                    # Get indices from position to end
                                    indices = np.arange(pos, len(diameter_init))

                                    np.put(m_ramales[i]["Material"], indices, "HA")
                                    np.put(m_ramales[i]["Seccion"], indices, "rectangular")
                                    np.put(m_ramales[i]["Rug"], indices, self.roughness.get_roughness_natural_iterable(['HA']))

                                    #include ramal to redo list, so it can be resized for rectangular part
                                    self.redo_ramales_list.append(i)


                            cond= False
                            continue


                        # Update the indices of the section where the material was found
                        index_insert = index_seccion[index_material]
                        # Update the internal diameters in the main array with the calculated diameters
                        np.put(m_ramales[i]["D_int"], index_insert, diametro_interno)

    def get_equivalent_diameter(self, diametro_in, material_out):
        """
        Convierte diámetro(s) de entrada al tamaño estándar más cercano disponible en el material de destino.

        Esta función busca en los tamaños estándar del material de salida y selecciona el que tenga
        la relación más cercana a 1.0 respecto al diámetro de entrada.

        Parameters
        ----------
        diametro_in: float or array-like

        material_out: str or array-like

        Returns
        -------
        float or np.ndarray
            Diámetro(s) estándar más cercano(es) en el material de salida.

        Examples
        --------


        Notas
        -----

        """
        if not isinstance(diametro_in, (list, np.ndarray)):
            # array diametro externo
            d_ext_array_out = self.get_internal_diameter_by_material(material_out)
            # get closet size in material_out from diametro_in in material_in
            closet_index = np.argmin(np.abs((diametro_in / d_ext_array_out) - 1))
            # D_out size
            D_out = d_ext_array_out[closet_index]
            return D_out
        else:
            D_out_array = []
            for diametro_in_pos, material_out_pos in zip(diametro_in, material_out):
                # array diametro externo
                d_ext_array_out = self.get_internal_diameter_by_material(material_out_pos)
                # get closet size in material_out from diametro_in in material_in
                closet_index = np.argmin(np.abs((diametro_in_pos / d_ext_array_out) - 1))
                # D_out size
                D_out_array.append(d_ext_array_out[closet_index])
            return np.array(D_out_array)

    @staticmethod
    def analogar_diametro_interno(diameter_dict, materials, new_outside_diameters):
        """
        Efficiently fits a polynomial to the diameter data for each unique material and predicts inside diameters for new outside diameters.

        :param diameter_dict: Dictionary of diameter data for different materials.
        :param materials: Array of materials corresponding to the new outside diameters.
        :param new_outside_diameters: Array of new outside diameters to predict inside diameters for.
        :return: Array of predicted inside diameters.
        """
        new_outside_diameters = new_outside_diameters.astype(float)
        predictions = np.empty(len(new_outside_diameters))
        unique_materials = np.unique(materials)

        for material in unique_materials:
            # Indices where current material is present
            index_material = np.where(materials == material)[0]

            # Extracting diameter data for the material
            diameter_data = diameter_dict.get(material, None)
            if diameter_data is None:
                predictions[index_material] = np.nan  # Material not found
                continue

            # Extracting outside and inside diameters from the dictionary
            outside_diameters = np.array(list(diameter_data.keys())).astype(float)
            inside_diameters = np.array(list(diameter_data.values())).astype(float)

            # Fitting a polynomial (3rd degree) to the data
            coefficients = np.polyfit(outside_diameters, inside_diameters, 3)
            polynomial = np.poly1d(coefficients)

            # Valid range for diameters
            min_value, max_value = np.min(outside_diameters), np.max(outside_diameters)
            min_value, max_value = float(min_value), float(max_value)
            # Vectorized prediction for all diameters of the current material
            valid_indices = (new_outside_diameters[index_material] >= min_value) & (
                        new_outside_diameters[index_material] <= max_value)
            invalid_indices = ~valid_indices

            # Predicting inside diameters
            predictions[index_material[valid_indices]] = polynomial(
                new_outside_diameters[index_material[valid_indices]])
            predictions[index_material[invalid_indices]] = np.nan  # Outside diameter out of range

            predictions[0] = predictions[1]  # to avoid error in sizing
        return np.round(predictions, 2)

    @staticmethod
    def get_internal_diameter_by_material(material):
        return  np.asarray(sorted(geometry_vg.diametro_interno_externo_pypiper[material].keys()))

    @staticmethod
    def get_external_diameter_from_internal(diameter_int, material):
        """

        :param diameter_int:
        :param material:
        :return: external diameter from internal diameter and material
        """
        return itemgetter(*diameter_int)(geometry_vg.diametro_interno_externo_pypiper[material])

    def insertar_d_int_no_decreciente(self, m_ramales, ramal_hacia):
        """
        Asegura que la dimensión vertical (D_int) en el ramal destino (`ramal_hacia`)
        no decrezca en la dirección del flujo, propagando el valor máximo acumulado
        desde el inicio del ramal .

        Para cada tipo de sección presente (circular, rectangular, trapezoidal, etc.),
        reconstruye el string de `D_int` usando la dimensión vertical no decreciente,
        manteniendo los otros parámetros geométricos (base, taludes) intactos.

        Args:
            m_ramales (dict): Diccionario con la geometría de los ramales.
            ramal_hacia (str): Clave del ramal destino cuyo D_int será corregido.

        Modifies:
            m_ramales[ramal_hacia]['D_int']: Actualizado con valores no decrecientes,
            formateados según el tipo de sección.
        """
        # Paso 1: Calcular la dimensión vertical no decreciente desde el ramal de origen
        dimension_vertical_no_decreciente = self.section_str2float(m_ramales[ramal_hacia]["D_int"]).copy()
        # dimension_vertical_no_decreciente = np.maximum.accumulate(dimension_vertical_no_decreciente, out=dimension_vertical_no_decreciente)

        # Paso 2: Extraer componentes numéricos del D_int actual del ramal destino
        dimensiones = self.section_str2float(m_ramales[ramal_hacia]['D_int'], return_all=True)
        base = dimensiones[:, 0]  # Columna 0: base
        talud_izq = dimensiones[:, 2]  # Columna 2: talud izquierdo
        talud_der = dimensiones[:, 3]  # Columna 3: talud derecho

        # Paso 3: Obtener array de tipos de sección (todos los campos tienen misma longitud)
        secciones = m_ramales[ramal_hacia]['Seccion']

        # Paso 4: Crear array de salida con el dtype definido en el esquema de geometría
        dtype_d_int = geometry_vg.type_values_dict['D_int']
        d_int_no_decreciente = np.empty_like(m_ramales[ramal_hacia]['D_int'], dtype=dtype_d_int)

        # Paso 5: Procesar solo los tipos de sección que realmente aparecen en este ramal
        secciones_unicas = np.unique(secciones)

        for tipo in secciones_unicas:
            mask = secciones == tipo
            dim = dimension_vertical_no_decreciente[mask]
            dim = np.maximum.accumulate(dim)

            if tipo == 'circular':
                # equivalencia de diámetros entre dos materiales diferentes
                dim = self.get_equivalent_diameter(dim, m_ramales[ramal_hacia]["Material"][mask])

                # Formato: "diametro" con 3 decimales (ej. "1.250")
                d_int_no_decreciente[mask] = np.char.mod('%.3f', dim)

            elif tipo in ('rectangular', 'rectangular_abierta'):
                # Formato: "base x calado" (ej. "1.2x0.8")
                b = base[mask]
                #This assumes a rectangular section with b = 2y for hydraulic efficiency
                d_int_no_decreciente[mask] = np.char.add(
                    np.char.mod('%.1f',  b ),
                    np.char.add('x', np.char.mod('%.1f', dim))
                )

            elif tipo == 'trapezoidal':
                # Formato: "base x calado x talud_izq x talud_der" (taludes sin decimales)
                b = base[mask]
                # Use efficient trapezoidal base: b ≈ (2 / √3) * dim ≈ 1.1547 * dim
                talud_optimo = geometry_vg.parametros_basicos['trapezoidal']['talud_optimo']  # ≈1.732

                ti = talud_izq[mask]
                td = talud_der[mask]
                # Use existing slopes if valid (>0 and not NaN), else optimal
                ti = np.where(np.isfinite(ti) & (ti > 0), ti, talud_optimo)
                td = np.where(np.isfinite(td) & (td > 0), td, talud_optimo)

                d_int_no_decreciente[mask] = (
                        np.char.mod('%.1f', b) + 'x' +
                        np.char.mod('%.1f', dim) + 'x' +
                        np.char.mod('%.0f', ti) + 'x' +
                        np.char.mod('%.0f', td)
                )

            elif tipo == 'triangular':
                # Formato: "0 x calado x talud_izq x talud_der"
                talud_optimo = geometry_vg.parametros_basicos['triangular']['talud_optimo']  # ≈1.0
                ti = talud_izq[mask]
                td = talud_der[mask]
                # Use existing slopes if valid (>0 and not NaN), else optimal
                ti = np.where(np.isfinite(ti) & (ti > 0), ti, talud_optimo)
                td = np.where(np.isfinite(td) & (td > 0), td, talud_optimo)

                d_int_no_decreciente[mask] = (
                        '0x' +
                        np.char.mod('%.1f', dim) + 'x' +
                        np.char.mod('%.0f', ti) + 'x' +
                        np.char.mod('%.0f', td)
                )


        # Paso 6: Actualizar el campo D_int en el ramal destino
        m_ramales[ramal_hacia]['D_int'] = d_int_no_decreciente




        return m_ramales

    # ------------------------------------------------------------------------------------------------------------------
    def verify_pipe_diameter_non_decreasing(self, m_ramales, xy_inter):
        """
        Verifica que en la dirección del flujo, los diámetros de los ramales
        no disminuyan. Es decir: D_salida >= D_entrada.

        conexionHacia (connections towards): A list of strings representing the downstream connections (e.g., "ramal.pz"), where flow is directed towards these points.
        conexionDesde (connections from): A list of lists, where each sublist contains upstream connections (e.g., ["ramal1.pz1", "ramal2.pz2"]) feeding into the corresponding conexionHacia.

        Parámetros
        ----------
        m_ramales : dict
            Estructura con los ramales y sus diámetros internos.

        """
        # Conexiones
        conexionHacia, conexionDesde = xy_inter

        # If there are no connections, there is nothing to verify.
        if  len(conexionHacia) == 0 and len(conexionDesde) == 0:
            return m_ramales

        s = pd.Series(conexionHacia, dtype=geometry_vg.type_values_dict['Pozo'])
        conexiones = s.str.split('.', expand=True)
        conexiones.columns=['ramal_hacia', 'pz_hacia']
        conexiones['conexionDesde'] = conexionDesde
        remove_diameter_int = {}

        for _, grupo in conexiones.groupby('ramal_hacia'):
            #revisar que todos los diámetros o secciones en conexionDesde sea menor o igual a la seccion corresponding en conexionHacia
            for conexionDesdeLista, ramal_hacia, pz_hacia in zip(grupo['conexionDesde'], grupo['ramal_hacia'], grupo['pz_hacia']):

                #diametro o seccion de conexionHacia
                try:
                    dimension_vertical_conexionHaciaItem = self.section_str2float(m_ramales[ramal_hacia]["D_int"][int(pz_hacia) + 1]).item()
                except:
                    dimension_vertical_conexionHaciaItem = self.section_str2float(m_ramales[ramal_hacia]["D_int"][int(pz_hacia)]).item()

                dimension_vertical_conexionDesdeItem = []
                seccion_conexionDesdeItem = []
                for conexionDesdeItem in conexionDesdeLista:
                    # parse conexionDesdeItem para ramal y position en ramal
                    ramal_desde, pz_desde = conexionDesdeItem.split('.')
                    #get dimension vertical
                    current_vertical_dimension =  self.section_str2float(m_ramales[ramal_desde]["D_int"][int(pz_desde)]).item()
                    current_section = m_ramales[ramal_desde]["D_int"][int(pz_desde)]
                    dimension_vertical_conexionDesdeItem.append(current_vertical_dimension)
                    seccion_conexionDesdeItem.append(current_section)

                # get maximum dimension in conexionDesde
                index_max = np.argmax(dimension_vertical_conexionDesdeItem)
                max_dimension_vertical_desde = dimension_vertical_conexionDesdeItem[index_max]
                max_seccion_desde = seccion_conexionDesdeItem[index_max]
                # verificar que la dimension vertical en conexionHacia sea mayor o igual a la dimension vertical maxima en conexionDesde
                cond_non_decreasing = dimension_vertical_conexionHaciaItem >= max_dimension_vertical_desde
                if not cond_non_decreasing:
                    ramal_pz =  '.'.join([ramal_hacia, pz_hacia ])
                    original_value = m_ramales[ramal_hacia]['D_min'][int(pz_hacia)]
                    remove_diameter_int[ramal_pz] =  original_value

                    try:
                        m_ramales[ramal_hacia]['D_min'][int(pz_hacia) + 1] = max_seccion_desde
                    except Exception as e:
                        m_ramales[ramal_hacia]['D_min'][int(pz_hacia)] = max_seccion_desde

                    try:
                        # asegurar diametro no decreciente en ramal_hacia,
                        m_ramales = self.get_sizing_int(m_ramales, xy_inter, ramal_hacia)
                    except:
                        m_ramales = self.get_sizing_int(m_ramales, xy_inter, ramal_hacia)
                        print()

            # asegurar diametro no decreciente en ramal_hacia,
            m_ramales = self.insertar_d_int_no_decreciente(m_ramales, ramal_hacia)


        if len(self.redo_ramales_list) > 0:
            redo_ramales_set = set(self.redo_ramales_list)
            for ramal in redo_ramales_set:
                m_ramales = self.section_sizing_int(m_ramales, [ramal])
            self.redo_ramales_list = []

        #update D_min to original values
        for ramal_pz, original_value in remove_diameter_int.items():
            ramal, pz = ramal_pz.split('.')
            m_ramales[ramal]['D_min'][int(pz)] = original_value


        return m_ramales

    #------------------------------------------------------------------------------------------------------------------
    def section_sizing_int(self, m_ramales, lista_ramales):

        # dimensionamiento de secciones
        for i in lista_ramales:
            # modificar valores de caudal cero para evitar errores en la minimización
            q_accu = m_ramales[i]["q_accu"].copy()
            q_accu = np.where(q_accu <= 0, 0.01, q_accu)

            # check for new ramales
            index_new = np.char.equal(m_ramales[i]["Estado"], "nuevo")
            index_existing = np.char.equal(m_ramales[i]["Estado"], "existente")

            # get int size for new sections
            if index_new.any() > 0:
                # indexar tipos de seccion
                secciones = np.unique(m_ramales[i]["Seccion"])

                empty_array = np.zeros(shape=len(m_ramales[i]["Estado"]), dtype=geometry_vg.type_values_dict['D_int'])
                m_ramales[i]["D_int"] = empty_array.copy()
                for seccion in secciones:
                    index_seccion = np.char.equal(m_ramales[i]["Seccion"], seccion).nonzero()[0]

                    pendiente =  m_ramales[i]["S"][index_seccion]
                    # manning =  m_ramales[i]["Rug"][index_seccion]
                    # material = m_ramales[i]["Material"][index_seccion]
                    q = q_accu[index_seccion]

                    # verificar que la pendiente no sea negativa
                    cond_pendiente = (pendiente <= 0).any()
                    if cond_pendiente:
                        print("error", i)
                        sys.exit("Hay pendientes negativas def -> get_sizing_int")


                    if seccion in ["circular"]:
                        # calcular diametro inicial con un valor de h/D = 0.75
                        diametro_init = self.spll.seccion_init_circular(q, m_ramales[i]["Rug"][index_seccion], pendiente)

                        nan_check = np.isnan(diametro_init).any()
                        if nan_check:
                            print("error", i)
                            sys.exit("Hay valores Nan en seccion circular def -> get_sizing_int")

                        # diametro minimo segun requerimientos de usuario
                        diametro_minimo_usuario =  self.section_str2float(m_ramales[i]["D_min"][index_seccion])
                        diametro_minimo_usuario =np.nan_to_num(diametro_minimo_usuario)

                        # diametro minimo segun parametros basicos
                        diametro_minimo = np.where(diametro_minimo_usuario > 0, diametro_minimo_usuario, geometry_vg.parametros_basicos['circular']['dimension_minima'])

                        # verificación que el diametro inicial no sea menor al diametro minimo
                        diametro_init = np.where(diametro_init <= diametro_minimo, diametro_minimo, diametro_init)

                        # verificación que el diametro vaya de mayor a menor
                        m_ramales[i]["D_int"][index_seccion] = diametro_init

                        # seleccionar diametro interno
                        self.select_internal_diameter(m_ramales, i, index_seccion, "circular")

                        #update index_secccion after select_internal_diameter, in case some section was changed to rectangular
                        index_seccion = np.char.equal(m_ramales[i]["Seccion"], seccion).nonzero()[0]

                        # verificación de que los diámetros del ramal vayan de menor a mayor después del dimensionamiento interno
                        diametro_no_decreciente = m_ramales[i]["D_int"][index_seccion].astype(float)
                        diametro_interno = np.maximum.accumulate(diametro_no_decreciente)

                        # equivalencia de diámetros entre dos materiales diferentes
                        diametro_interno_homologado = self.get_equivalent_diameter(diametro_interno, m_ramales[i]["Material"][index_seccion])

                        # asignar diámetros a array de diametro interno
                        m_ramales[i]["D_int"][index_seccion] = diametro_interno_homologado


                    elif seccion in ["rectangular", 'rectangular_abierta']:
                        # dimensionar base y altura inicial de canal rectangular¡
                        b, y = self.spll.seccion_init_rectangular(i, q, m_ramales[i]["Rug"][index_seccion], pendiente)


                        nan_check = np.isnan(y).any()
                        if nan_check:
                            print("error", i)
                            sys.exit("Hay valores Nan en seccion rectangular def -> get_sizing_int")

                        # redondear a múltiplos de 0.05m
                        y = np.around(y / 0.05) * 0.05

                        # dimensiones minimas según requerimientos de usuario
                        dimensiones_minimas_usuarios= self.section_str2float(m_ramales[i]["D_min"][index_seccion], return_all=True)
                        b_minima_usuario = np.nan_to_num(dimensiones_minimas_usuarios[:, 0])
                        y_minima_usuario = np.nan_to_num(dimensiones_minimas_usuarios[:, 1])

                        # verificación que la dimension inicial no sea menor a dimension vertical minima del usuario
                        y = np.where(y >= y_minima_usuario, y, y_minima_usuario)
                        b = np.where(b >= b_minima_usuario, b, b_minima_usuario)

                        # verificación de que los diámetros del ramal vayan de menor a mayor después del dimensionamiento interno
                        y_no_decreciente = np.maximum.accumulate(y)
                        b_no_decreciente = np.maximum.accumulate(b)

                        # join bxy
                        rectangular_seccion = (
                                np.char.add(np.char.mod('%.2f', b_no_decreciente ),
                                np.char.add('x',
                                np.char.mod('%.2f', y_no_decreciente)))
                        )

                        # asignar secciones a array de diametro interno
                        m_ramales[i]["D_int"][index_seccion] = rectangular_seccion


                    elif seccion in ["trapezoidal"]:
                        # dimensionar base y altura inicial de canal rectangular
                        b, y, talud_izquierdo, talud_derecho = self.spll.seccion_init_trapezoidal(q, m_ramales[i]["Rug"][index_seccion], pendiente)

                        nan_check = np.isnan(y).any()
                        if nan_check:
                            print("error", i)
                            sys.exit("Hay valores Nan en seccion rectangular def -> get_sizing_int")

                        # redondear a múltiplos de 0.05m
                        y = np.around(y / 0.05) * 0.05

                        # dimensiones minimas según requerimientos de usuario
                        dimensiones_minimas_usuarios= self.section_str2float(m_ramales[i]["D_min"][index_seccion], return_all=True)
                        b_minima_usuario = np.nan_to_num(dimensiones_minimas_usuarios[:, 0])
                        y_minima_usuario = np.nan_to_num(dimensiones_minimas_usuarios[:, 1])
                        talud_izquierdo_usuario = dimensiones_minimas_usuarios[:, 2]
                        talud_derecho_usuario = dimensiones_minimas_usuarios[:, 3]

                        # verificación que la dimension inicial no sea menor a dimension vertical minima del usuario
                        y = np.where(y >= y_minima_usuario, y, y_minima_usuario)
                        b = np.where(b >= b_minima_usuario, b, b_minima_usuario)
                        talud_izquierdo = np.where(np.isnan(talud_izquierdo_usuario), talud_izquierdo, talud_izquierdo_usuario)
                        talud_derecho = np.where(np.isnan(talud_derecho_usuario), talud_derecho, talud_derecho_usuario)

                        # verificación de que las dimensiones del ramal vayan de menor a mayor después del dimensionamiento interno
                        y_no_decreciente = np.maximum.accumulate(y)
                        b_no_decreciente = np.maximum.accumulate(b)

                        # Formato: b x y x m_izq x m_der
                        trapezoidal_seccion = np.char.add(
                            np.char.add(
                                np.char.add(
                                    np.char.mod('%.2f', b_no_decreciente),
                                    np.char.add('x', np.char.mod('%.2f', y_no_decreciente))
                                ),
                                np.char.add('x', np.char.mod('%.2f', talud_izquierdo))
                            ),
                            np.char.add('x', np.char.mod('%.2f', talud_derecho))
                        )

                        # asignar secciones a array de diametro interno
                        m_ramales[i]["D_int"][index_seccion] = trapezoidal_seccion


                    elif seccion in ["triangular"]:
                        # dimensionar base y altura inicial de canal rectangular
                        y, talud_izquierdo, talud_derecho = self.spll.seccion_init_triangular(q, m_ramales[i]["Rug"][index_seccion], pendiente)

                        nan_check = np.isnan(y).any()
                        if nan_check:
                            print("error", i)
                            sys.exit("Hay valores Nan en seccion rectangular def -> get_sizing_int")

                        # redondear a múltiplos de 0.05m
                        y = np.around(y / 0.05) * 0.05

                        # dimensiones minimas según requerimientos de usuario
                        dimensiones_minimas_usuarios= self.section_str2float(m_ramales[i]["D_min"][index_seccion], return_all=True)
                        y_minima_usuario = np.nan_to_num(dimensiones_minimas_usuarios[:, 1])
                        talud_izquierdo_usuario = dimensiones_minimas_usuarios[:, 2]
                        talud_derecho_usuario = dimensiones_minimas_usuarios[:, 3]

                        # verificación que la dimension inicial no sea menor a dimension vertical minima del usuario
                        y = np.where(y >= y_minima_usuario, y, y_minima_usuario)
                        talud_izquierdo = np.where(np.isnan(talud_izquierdo_usuario), talud_izquierdo, talud_izquierdo_usuario)
                        talud_derecho = np.where(np.isnan(talud_derecho_usuario), talud_derecho, talud_derecho_usuario)

                        # verificación de que las dimensiones del ramal vayan de menor a mayor después del dimensionamiento interno
                        y_no_decreciente = np.maximum.accumulate(y)

                        # Formato: b x y x m_izq x m_der
                        triangular_seccion = np.char.add(
                            np.char.add(
                                np.char.add(
                                    np.char.mod('%.2f', np.full(y_no_decreciente.size, fill_value=0)),
                                    np.char.add('x', np.char.mod('%.2f', y_no_decreciente))
                                ),
                                np.char.add('x', np.char.mod('%.2f', talud_izquierdo))
                            ),
                            np.char.add('x', np.char.mod('%.2f', talud_derecho))
                        )

                        # asignar secciones a array de diametro interno
                        m_ramales[i]["D_int"][index_seccion] = triangular_seccion


                    else:
                        print('error: no se reconoce el tipo de sección en def -> section_sizing_int')
                        m_ramales[i]["D_int"][index_seccion] =  np.nan

            # get int size for existing sections
            if index_existing.any() > 0:
                # indexar tipos de seccion
                secciones = np.unique(m_ramales[i]["Seccion"])
                empty_array = np.zeros(shape=len(m_ramales[i]["Estado"]), dtype=geometry_vg.type_values_dict['D_int'])
                m_ramales[i]["D_int"] = empty_array.copy()

                for seccion in secciones:
                    index_seccion = np.char.equal(m_ramales[i]["Seccion"], seccion).nonzero()[0]

                    if seccion in ["circular"]:
                        m_ramales[i]["D_ext"][0] = m_ramales[i]["D_ext"][1]

                        # analogar diametro interno del ramal existente
                        diametro_init = self.analogar_diametro_interno(
                            geometry_vg.diametro_interno_externo_pypiper,
                            m_ramales[i]["Material"][index_seccion],
                            m_ramales[i]["D_ext"][index_seccion],
                        )

                        # fill nan values
                        filtro_nan = np.isnan(diametro_init)
                        diametro_init[filtro_nan.nonzero()] = m_ramales[i]["D_ext"][filtro_nan.nonzero()].astype(float)

                        # no se hace corrections porque son diámetros existentes
                        m_ramales[i]["D_int"][index_seccion] = diametro_init

                    elif seccion in ["rectangular", 'rectangular_abierta', 'trapezoidal', 'triangular']:

                        # set internal section
                        m_ramales[i]["D_int"][index_seccion] = m_ramales[i]["D_ext"][index_seccion]

                    else:
                        m_ramales[i]["D_int"][index_seccion] = np.nan

        return m_ramales

    # ------------------------------------------------------------------------------------------------------------------
    def get_sizing_int(self, m_ramales, xy_inter, ramal_value=None):


        # check for specific ramal
        if ramal_value:
            lista_ramales = [ramal_value]
        else:
            lista_ramales = m_ramales.keys()

        # section sizing
        m_ramales = self.section_sizing_int(m_ramales, lista_ramales)

        # verify non decreasing diameter
        m_ramales = self.verify_pipe_diameter_non_decreasing(m_ramales, xy_inter)

        return m_ramales

    # ------------------------------------------------------------------------------------------------------------------
    def get_sizing_ext(self, m_ramales, ramal_value=None):

        if ramal_value:
            lista_ramales = [ramal_value]
        else:
            lista_ramales = m_ramales.keys()

        for i in lista_ramales:
            # índice para modificar los tramos nuevos
            index = np.char.equal(m_ramales[i]["Estado"], "nuevo").nonzero()[0]

            if index.size > 0:
                m_ramales[i]["D_ext"] = np.zeros(shape=len(m_ramales[i]["D_int"])).astype(geometry_vg.type_values_dict['D_ext'])
                secciones_array = m_ramales[i]["Seccion"].copy()
                secciones = np.unique(secciones_array)

                for seccion in secciones:
                    index_seccion = np.char.equal(secciones_array, seccion).nonzero()[0]
                    materiales = set(m_ramales[i]["Material"][index_seccion])

                    if seccion in ["circular"]:
                        for material in materiales:
                            index_material = np.where(m_ramales[i]["Material"][index_seccion] == material)[0]
                            index_seccion_material = index_seccion[index_material]

                            seccion_interior = self.section_str2float(m_ramales[i]["D_int"][index_seccion_material])
                            # # analogar diametros internos con los existentes en el material
                            # seccion_interior = self.get_equivalent_diameter(seccion_interior, m_ramales[i]["Material"][index_seccion_material])
                            #obtener el diametro externo
                            seccion_exterior = self.get_external_diameter_from_internal(seccion_interior, material)


                            np.put(m_ramales[i]["D_ext"], index_seccion_material, seccion_exterior)

                    elif seccion in ["rectangular", 'rectangular_abierta', 'trapezoidal', 'triangular']:
                        for material in materiales:
                            index_material = np.where(m_ramales[i]["Material"][index_seccion] == material)[0]
                            index_seccion_material = index_seccion[index_material]

                            seccion_interior = m_ramales[i]["D_int"][index_seccion_material]
                            np.put(m_ramales[i]["D_ext"], index_seccion_material, seccion_interior)


        return m_ramales

    # ------------------------------------------------------------------------------------------------------------------
    def get_sizing_SPLL(self, m_ramales, xy_inter):

        # get Froude values
        m_ramales = self.hydraulic_conditions.get_hydraulic_conditions(m_ramales)

        # ubicar tramos con h/d >= valor predeterminado
        h_D_dict = self.get_h_D_dict(m_ramales)

        cont = 0
        while len(h_D_dict) > 0 and cont < 100:
            for i in h_D_dict.keys():
                # check for new and existing ramales
                index_new = np.char.equal(m_ramales[i]["Estado"], "nuevo").nonzero()[0]
                index_existing = np.char.equal(m_ramales[i]["Estado"], "existente").nonzero()[0]

                if index_new.size > 0:
                    # indexar tipos de seccion
                    secciones = np.unique(m_ramales[i]["Seccion"])
                    for seccion in secciones:
                        index_seccion = np.char.equal(m_ramales[i]["Seccion"], seccion).nonzero()[0]

                        if seccion in ["circular"]:
                            # inicializar bandera de control de cambio de sección
                            seccion_control = False

                            # obtener el conjunto único de materiales presentes en los tramos de esta sección
                            materiales = set(m_ramales[i]["Material"][index_seccion])
                            for material in materiales:
                                # seleccionar índices de todos los tramos que tienen este mismo material
                                index_material = np.where(m_ramales[i]["Material"] == material)[0]

                                # calcular intersección de índices: tramos con h/D alto, que pertenecen a la sección actual y al material actual
                                # reduce(np.intersect1d, [...]) devuelve índices que cumplen las tres condiciones simultáneamente
                                index_change = reduce(np.intersect1d, [h_D_dict[i], index_seccion, index_material])

                                # cuando se cambia material y seccion este índice queda vacío, entonces si pasa esto solo se continúa el for loop
                                if len(index_change) == 0:
                                    continue
                                # obtener la lista de diámetros internos disponibles para este material (ordenada)
                                diametro_interno_material = self.get_internal_diameter_by_material(material)

                                # copiar los diámetros actuales de los tramos que deben modificarse
                                diametro_previo = m_ramales[i]["D_int"][index_change].copy()

                                # buscar la posición donde insertar cada diámetro previo en la lista de diámetros estándar
                                # np.searchsorted devuelve el índice para mantener el orden; +1 selecciona el siguiente mayor (tamaño superior)
                                index_search = np.searchsorted(diametro_interno_material, diametro_previo) + 1

                                # si todos los índices buscados están dentro del rango de tamaños estándar disponibles
                                if np.max(index_search) < diametro_interno_material.size:
                                    # seleccionar los diámetros estándar correspondiente (tamaños superiores)
                                    diametro_modificado = diametro_interno_material[index_search]
                                    # asignar los nuevos diámetros a las posiciones originales
                                    np.put(m_ramales[i]["D_int"], index_change, diametro_modificado)
                                else:
                                    # Identify the local positions (within index_change) that correspond to oversized pipes.
                                    oversized_local_indices = np.where(index_search >= diametro_interno_material.size)[0]

                                    # Get the global indices in the branch array for the sections that need to be converted.
                                    indices_to_convert = index_change[oversized_local_indices]

                                    # Filter out the index 0, if present, to avoid modifying the starting point of the branch incorrectly.
                                    indices_to_convert = indices_to_convert[indices_to_convert != 0]

                                    # If a previous diameter exceeds the largest available size for this material,
                                    # convert those sections to closed rectangular and change the material to "HA".
                                    print(f"Changing circular section to closed rectangular in ramal: {i}: {indices_to_convert}")

                                    if len(indices_to_convert) > 0:
                                        # Change the material of these sections to "HA" (Hormigón Armado - Reinforced Concrete).
                                        m_ramales[i]["Material"][indices_to_convert] = "HA"
                                        # Change the section type to "rectangular".
                                        m_ramales[i]["Seccion"][indices_to_convert] = "rectangular"
                                        m_ramales[i]["Rug"][indices_to_convert] = self.roughness.get_roughness_natural_iterable(['HA'])

                                        # Convert the internal and external section format to rectangular.
                                        # The new rectangular dimension is based on the previous diameter.
                                        new_rect_dimension = self.section_str2float(m_ramales[i]["D_int"][indices_to_convert])

                                        # Create the new rectangular dimension string "dimension x dimension".
                                        rectangular_section_str = np.char.add(
                                            np.char.mod('%.1f', new_rect_dimension),
                                            np.char.add('x', np.char.mod('%.1f', new_rect_dimension))
                                        )
                                        m_ramales[i]["D_int"][indices_to_convert] = rectangular_section_str
                                        m_ramales[i]["D_ext"][indices_to_convert] = rectangular_section_str


                                        # Set the fill angle to 0 to indicate it's no longer a circular section.
                                        m_ramales[i]["ang"][oversized_local_indices] = 0

                                        # Mark that a structural change occurred, so the loop will break and re-evaluate.
                                        seccion_control = True
                                        break

                            # si se realizó un cambio de sección/material que requiere re-evaluar, salir del bucle de materiales
                            if seccion_control:
                                break

                            # verificación: asegurar que la secuencia de diámetros a lo largo del ramal no disminuya en sentido de flujo
                            # convertir a float y tomar el arreglo de diámetros actuales para la sección
                            d_init_array = m_ramales[i]["D_int"][index_seccion].astype(float)

                            # aplicar acumulado máximo para forzar no-decreciente: cada elemento es al menos el máximo anterior
                            d_int_no_decreciente = np.maximum.accumulate(d_init_array, out=d_init_array)

                            # transformar / mapear los diámetros internos a los diámetros estándar
                            # disponibles por material (equivalencia entre materiales)
                            d_int_translate = self.get_equivalent_diameter(d_int_no_decreciente, m_ramales[i]["Material"][index_seccion])

                            # asignar los diámetros equivalentes finales al arreglo global D_int para la sección
                            m_ramales[i]["D_int"][index_seccion] = d_int_translate

                        elif seccion in ["rectangular", "rectangular_abierta"]:
                            # calcular intersección de índices: tramos con h/D alto, que pertenecen a la sección actual y al material actual
                            index_change = np.intersect1d(h_D_dict[i], index_seccion)

                            if len(index_change) > 0:
                                #hallar las dimensiones actuales de los tramos que deben modificarse de canal rectangular
                                dimensiones = self.section_str2float(m_ramales[i]["D_int"][index_change], return_all=True)
                                base = dimensiones[:, 0]
                                calado_max = dimensiones[:, 1]
                                pendiente = m_ramales[i]["S"][index_change]
                                rugosidad = m_ramales[i]["Rug"][index_change]
                                q_target = m_ramales[i]["q_accu"][index_change]

                                all_args = [base, pendiente, rugosidad, q_target]
                                root_func = self.spll.rectangular_spll_root

                                # Find water depth for prismatic section
                                calado = self.spll._find_roots(
                                    root_func=root_func,
                                    ramal_id=i,
                                    all_args=all_args,
                                    lower_bound=0.001,
                                    upper_bound=calado_max.max() * 2,
                                )
                                calado = np.round(calado, 3)


                                froude_number = pd.Series(m_ramales[i]["Froude"][index_change]).map(geometry_vg.map_calado_dict).to_numpy()
                                factor = 1 + (1 - froude_number)
                                # nuevo seccion de canal (h)
                                calado_propuesto = np.round(calado * factor / 0.05) * 0.05 + 0.10
                                # Verificar que calado sea al menos calado_propuesto (redondeado a 0.05m)
                                calado = np.maximum(calado_max, calado_propuesto)
                                #evitar un bucle infinito cuando se dimensiona muy cerca del calado maximo
                                calado =  np.where(np.abs(calado - calado_max) < 0.01, calado_max + 0.05, calado)

                                # verificar que vaya de menor a mayor la dimension h y b
                                calado = np.maximum.accumulate(calado)
                                base = np.maximum.accumulate(base)

                                # join BxY
                                rectangular_seccion = np.char.add(
                                    np.char.mod('%.1f', base),
                                    np.char.add('x', np.char.mod('%.1f', calado))
                                )

                                # set internal section
                                m_ramales[i]["D_int"][index_change] = rectangular_seccion

                        elif seccion in ["trapezoidal"]:
                            # calcular intersección de índices: tramos con h/D alto, que pertenecen a la sección actual y al material actual
                            index_change = np.intersect1d(h_D_dict[i], index_seccion)

                            if len(index_change) > 0:
                                #hallar las dimensiones actuales de los tramos que deben modificarse de canal rectangular
                                dimensiones = self.section_str2float(m_ramales[i]["D_int"][index_change], return_all=True)
                                base = dimensiones[:, 0]
                                calado_max = dimensiones[:, 1]
                                talud_izquierdo= dimensiones[:, 2]
                                talud_derecho = dimensiones[:, 3]
                                pendiente = m_ramales[i]["S"][index_change]
                                rugosidad = m_ramales[i]["Rug"][index_change]
                                q_target = m_ramales[i]["q_accu"][index_change]

                                all_args = [base, pendiente, rugosidad, q_target, talud_izquierdo, talud_derecho]
                                root_func = self.spll.trapezoidal_spll_root

                                # Find water depth for prismatic section
                                calado = self.spll._find_roots(
                                    root_func=root_func,
                                    ramal_id=i,
                                    all_args=all_args,
                                    lower_bound=0.001,
                                    upper_bound=calado_max.max() * 2,
                                )
                                calado = np.round(calado, 3)


                                froude_number = pd.Series(m_ramales[i]["Froude"][index_change]).map(geometry_vg.map_calado_dict).to_numpy()
                                factor = 1 + (1 - froude_number)
                                # nuevo seccion de canal (h)
                                calado_propuesto = np.round(calado * factor / 0.05) * 0.05 + 0.10
                                # Verificar que calado sea al menos calado_propuesto (redondeado a 0.05m)
                                calado = np.maximum(calado_max, calado_propuesto)
                                #evitar un bucle infinito cuando se dimensiona muy cerca del calado maximo
                                calado =  np.where(np.abs(calado - calado_max) < 0.01, calado_max + 0.05, calado)

                                # verificar que vaya de menor a mayor la dimension h y b
                                calado = np.maximum.accumulate(calado)
                                base = np.maximum.accumulate(base)

                                # Formato: b x y x m_izq x m_der
                                trapezoidal_seccion = np.char.add(
                                    np.char.add(
                                        np.char.add(
                                            np.char.mod('%.2f', base),
                                            np.char.add('x', np.char.mod('%.2f', calado))
                                        ),
                                        np.char.add('x', np.char.mod('%.2f', talud_izquierdo))
                                    ),
                                    np.char.add('x', np.char.mod('%.2f', talud_derecho))
                                )


                                # set internal section
                                m_ramales[i]["D_int"][index_change] = trapezoidal_seccion

                        elif seccion in ["triangular"]:
                            # calcular intersección de índices: tramos con h/D alto, que pertenecen a la sección actual y al material actual
                            index_change = np.intersect1d(h_D_dict[i], index_seccion)

                            if len(index_change) > 0:
                                #hallar las dimensiones actuales de los tramos que deben modificarse de canal rectangular
                                dimensiones = self.section_str2float(m_ramales[i]["D_int"][index_change], return_all=True)
                                base = dimensiones[:, 0]
                                calado_max = dimensiones[:, 1]
                                talud_izquierdo= dimensiones[:, 2]
                                talud_derecho = dimensiones[:, 3]
                                pendiente = m_ramales[i]["S"][index_change]
                                rugosidad = m_ramales[i]["Rug"][index_change]
                                q_target = m_ramales[i]["q_accu"][index_change]

                                all_args = [pendiente, rugosidad, q_target, talud_izquierdo, talud_derecho]
                                root_func = self.spll.triangular_spll_root

                                # Find water depth for prismatic section
                                calado = self.spll._find_roots(
                                    root_func=root_func,
                                    ramal_id=i,
                                    all_args=all_args,
                                    lower_bound=0.001,
                                    upper_bound=calado_max.max() * 2,
                                )
                                calado = np.round(calado, 3)


                                froude_number = pd.Series(m_ramales[i]["Froude"][index_change]).map(geometry_vg.map_calado_dict).to_numpy()
                                factor = 1 + (1 - froude_number)
                                # nuevo seccion de canal (h)
                                calado_propuesto = np.round(calado * factor / 0.05) * 0.05 + 0.10
                                # Verificar que calado sea al menos calado_propuesto (redondeado a 0.05m)
                                calado = np.maximum(calado_max, calado_propuesto)
                                #evitar un bucle infinito cuando se dimensiona muy cerca del calado maximo
                                calado =  np.where(np.abs(calado - calado_max) < 0.01, calado_max + 0.05, calado)

                                # verificar que vaya de menor a mayor la dimension h y b
                                calado = np.maximum.accumulate(calado)

                                # Formato: b x y x m_izq x m_der
                                triangular_seccion = np.char.add(
                                    np.char.add(
                                        np.char.add(
                                            np.char.mod('%.2f', np.full(calado.size, fill_value=0)),
                                            np.char.add('x', np.char.mod('%.2f', calado))
                                        ),
                                        np.char.add('x', np.char.mod('%.2f', talud_izquierdo))
                                    ),
                                    np.char.add('x', np.char.mod('%.2f', talud_derecho))
                                )

                                # set internal section
                                m_ramales[i]["D_int"][index_change] = triangular_seccion

                        else:
                            m_ramales[i]["D_int"][index_seccion] = np.nan
                            print('error: no se reconoce el tipo de sección en def get_sizing_SPLL')

                if index_existing.size > 0:
                    pass

            # verificar que los diametros de los afluentes sean menor o igual al efluente
            m_ramales = self.verify_pipe_diameter_non_decreasing(m_ramales, xy_inter)

            # "asignar diametros externos, caudal y velocidad seccion llena, dimensionamiento en seccion parcialmente llena"
            m_ramales = self.get_sizing_ext(m_ramales)
            m_ramales = self.sll.get_SLL(m_ramales)
            m_ramales = self.spll.get_SPLL(m_ramales)

            # ubicar tramos con h/d >= valor predeterminado
            h_D_dict = self.get_h_D_dict(m_ramales)

            cont = cont + 1

        if cont >= 100:
            print(h_D_dict)
            sys.exit("error def get_sizing_SPLL")

        return m_ramales

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_h_D_dict(m_ramales):
        is_close_percentage = 0.05

        h_D_dict = {}
        for i in m_ramales.keys():
            if "nuevo" in m_ramales[i]["Estado"]:
                Froude_value = m_ramales[i]["Froude"]  # Get the Froude value
                h_D_max = pd.Series(Froude_value).map(geometry_vg.map_calado_dict).to_numpy()  # Map Froude to h/D threshold
                # h_D_max = pd.Series(Froude_value).map({'': 0.35, 'critico': 0.35, 'sub-critico': 0.35, 'super-critico': 0.35}).to_numpy()  # Map Froude to h/D threshold

                h_D_values = m_ramales[i]["h/D"]
                condition = np.invert((h_D_values <= h_D_max) | (np.isclose(h_D_values, h_D_max, rtol=is_close_percentage)))

                h_D_dict[i] = np.where(condition)[0]

                # Only keep entries where there are any matching values
                if len(h_D_dict[i]) == 0:
                    del h_D_dict[i]

        return h_D_dict


class HydraulicConditions:
    """
    Clase para el análisis hidráulico de conducciones en flujo a superficie libre.
    Calcula parámetros fundamentales de diseño para cada tramo (ramal) de la red.

    Propósito:
    -----------
    Evaluar el comportamiento hidráulico (tensión tractiva, Froude, Reynolds y abrasión)
    en función de la geometría, pendiente, rugosidad y régimen de flujo.

    Dependencia:
    -------------
    Requiere la función `section_str2float()` para convertir cadenas de geometría
    a dimensiones numéricas (base, altura, taludes).
    """

    def __init__(self):
        # Constantes físicas universales
        self.gravedad = 9.81             # [m/s²]
        self.densidad_agua = 1000        # [kg/m³]
        self.kv = 1.004e-6               # [m²/s] viscosidad cinemática
        self.porcentaje_sedimentos = 0.15  # fracción sólida [-]

    # ---------------------------------------------------------------------
    @staticmethod
    def section_str2float(arr, return_b=False, return_all=False, sep="x"):

        r"""
        Converts an array of section dimension strings into a numeric array.

        The function parses strings that define geometric sections (circular, rectangular, trapezoidal or triangular)
         and returns specific dimensions as a float array.

        Dimension String Format:
        - Circular: "diameter" (e.g., "0.5")
        - Rectangular: "bottom base x height" (e.g., "2.0x1.5")
        - Trapezoidal: " top base x height x slope left x slope right" (e.g., "2.0x1.5x0.5x0.5")
        -Triangular: " 0 x height x slope left x slope right" (e.g., "0x1.5x0.25x0.25")

        Graphical representation:
                                                                                       ---
              ******            **************    ******************     ***********    │
           ************         *            *     ****************       *********     │
          **************        *            *      **************         *******      │  geom1
          **************        *            *       ************           *****       │
           ************         **************        **********             ***        │
              ******                                                          *         │
                                                                                       ---
          |----geom0---|        |----geom0---|        |-geom0-|            geom0 = 0

             Círculo              Rectángulo           Trapecio           Triángulo

        geom0: (circular:diámetro, rectangular:base, trapezoidal:base inferior, triangular: 0m)
        geom1: altura
        geom2: pendiente izquierda (vertical/horizontal)
        geom3: pendiente derecha (vertical/horizontal)

        """

        # crear series de pandas
        s = pd.Series(arr)

        # separar dimensiones de secciones
        dimensiones_df = s.str.split(sep, expand=True)

        dimensiones_array = dimensiones_df.to_numpy(dtype=float)
        l, dims = dimensiones_array.shape

        # Initialize geometry arrays
        geom = np.full((l, 4), np.nan)

        if dims == 1:
            geom[:, 0] = dimensiones_array[:, 0] #diametro

        elif dims == 2:
            geom[:, 0] = dimensiones_array[:, 0] # base
            geom[:, 1] = dimensiones_array[:, 1] # altura

        elif dims == 4:
            geom[:, 0] = dimensiones_array[:, 0]  # base
            geom[:, 1] = dimensiones_array[:, 1]  # altura
            geom[:, 2] = dimensiones_array[:, 2]  # talud izquierdo
            geom[:, 3] = dimensiones_array[:, 3]  # talud derecho

        else:
            pass

        if return_all:
            return geom

        #return specific dimension
        geom0 = geom[:, 0]
        geom1 = geom[:, 1]
        geom2 = geom[:, 2]
        geom3 = geom[:, 3]

        # get circular and prismatic indexes
        circular_index = (geom0 > 0) & np.isnan(geom1) & np.isnan(geom2) & np.isnan(geom3)
        prismatic_index = ~circular_index

        # zeros array
        out = np.zeros(shape=l, dtype=float)

        # Circular: return diameter
        out[circular_index] = geom0[circular_index]

        if return_b:
            base_superior_triangulo = (geom1 / geom2) + (geom1 / geom3)

            # return height for all prismatic sections
            out[prismatic_index] = np.where(geom0[prismatic_index] > 0, geom0[prismatic_index], base_superior_triangulo)
        else:
            # return height for all prismatic sections
            out[prismatic_index] = geom1[prismatic_index]

        return out

    # ---------------------------------------------------------------------
    def get_tension_tractiva(self, m_ramales, i):
        """
        Calcula la tensión tractiva (τ) sobre el perímetro mojado [Pa].
        τ = ρ * g * S * Rh
        """
        Seccion = m_ramales[i]["Seccion"]
        secciones = np.unique(Seccion)
        tension_tractiva = np.zeros(shape=len(Seccion))

        geom = self.section_str2float(m_ramales[i]["D_int"], return_all=True)
        base = geom[:, 0]
        talud_izquierdo = geom[:, 2]
        talud_derecho = geom[:, 3]

        for _seccion in secciones:
            seccion_index = np.char.equal(Seccion, _seccion).nonzero()[0]

            # Circular
            if _seccion == "circular":
                d_int = base[seccion_index]
                angulo = np.radians(m_ramales[i]["ang"][seccion_index])
                Rh = (1 - (np.sin(angulo) / angulo)) * (d_int / 4.0)
                tau = self.densidad_agua * self.gravedad * m_ramales[i]["S"][seccion_index] * Rh
                tension_tractiva[seccion_index] = tau

            # Prismática (rectangular, trapezoidal, triangular)
            elif _seccion in ["rectangular", "trapezoidal", "triangular"]:
                base_i = base[seccion_index]
                talud_i = talud_izquierdo[seccion_index]
                talud_d = talud_derecho[seccion_index]
                y = m_ramales[i]["h"][seccion_index]

                area = base_i * y + (talud_i + talud_d) * y**2
                perimetro = base_i + y * (
                    np.sqrt(1 + talud_i**2) + np.sqrt(1 + talud_d**2)
                )
                Rh = area / perimetro

                tau = self.densidad_agua * self.gravedad * m_ramales[i]["S"][seccion_index] * Rh
                tension_tractiva[seccion_index] = tau

        return tension_tractiva

    # ---------------------------------------------------------------------
    def get_froude_number(self, m_ramales, i):
        """
        Calcula el número de Froude (Fr) y clasifica el régimen del flujo.
        Fr = v / sqrt(g * h)
        """
        v = m_ramales[i]["v"]
        h = m_ramales[i]["h"]
        froude = v / np.sqrt(self.gravedad * h)

        froude_string = np.empty(shape=len(froude), dtype="U256")
        froude_string[np.where(froude < 1)[0]] = "sub-critico"
        froude_string[np.where(froude == 1)[0]] = "critico"
        froude_string[np.where(froude > 1)[0]] = "super-critico"

        return froude_string

    # ---------------------------------------------------------------------
    def get_reynolds_number(self, m_ramales, i):
        """
        Calcula el número de Reynolds (Re) y clasifica el régimen de flujo.
        Re = (v * Rh) / kv
        """
        Seccion = m_ramales[i]["Seccion"]
        secciones = np.unique(Seccion)
        reynolds = np.zeros(shape=len(Seccion))
        reynolds_string = np.empty(shape=len(reynolds), dtype="U256")

        geom = self.section_str2float(m_ramales[i]["D_int"], return_all=True)
        base = geom[:, 0]
        talud_izquierdo = geom[:, 2]
        talud_derecho = geom[:, 3]

        for _seccion in secciones:
            seccion_index = np.char.equal(Seccion, _seccion).nonzero()[0]

            # Circular
            if _seccion == "circular":
                d_int = base[seccion_index]
                angulo = np.radians(m_ramales[i]["ang"][seccion_index])
                Rh = (1 - (np.sin(angulo) / angulo)) * (d_int / 4.0)
                Re = m_ramales[i]["v"][seccion_index] * Rh / self.kv
                reynolds[seccion_index] = Re

            # Prismática
            elif _seccion in ["rectangular", "trapezoidal", "triangular"]:
                base_i = base[seccion_index]
                talud_i = talud_izquierdo[seccion_index]
                talud_d = talud_derecho[seccion_index]
                y = m_ramales[i]["h"][seccion_index]

                area = base_i * y + (talud_i + talud_d) * y**2
                perimetro = base_i + y * (
                    np.sqrt(1 + talud_i**2) + np.sqrt(1 + talud_d**2)
                )
                Rh = area / perimetro

                Re = m_ramales[i]["v"][seccion_index] * Rh / self.kv
                reynolds[seccion_index] = Re

        reynolds_string[np.where(reynolds < 2300)[0]] = "laminar"
        reynolds_string[np.intersect1d(
            np.where(reynolds >= 2300)[0], np.where(reynolds <= 2900)[0]
        )] = "transicion"
        reynolds_string[np.where(reynolds > 2900)[0]] = "turbulento"

        return reynolds_string

    # ---------------------------------------------------------------------
    def indice_abrasion(self, m_ramales, i):
        """
        Calcula un índice relativo de abrasión del conducto.
        Depende de la velocidad, calado, rugosidad y concentración de sedimentos.
        """
        return (
            m_ramales[i]["v"] * self.porcentaje_sedimentos * m_ramales[i]["h"]
        ) / (
            self.section_str2float(m_ramales[i]["D_ext"]) * m_ramales[i]["Rug"]
        )

    # ---------------------------------------------------------------------
    def get_hydraulic_conditions(self, m_ramales):
        """
        Evalúa todas las condiciones hidráulicas (τ, Fr, Re, abrasión)
        para cada tramo del diccionario m_ramales.
        """
        for i in m_ramales.keys():
            m_ramales[i]["Tension"] = self.get_tension_tractiva(m_ramales, i)
            m_ramales[i]["Froude"] = self.get_froude_number(m_ramales, i)
            m_ramales[i]["Reynolds"] = self.get_reynolds_number(m_ramales, i)
            m_ramales[i]["indice_abrasion"] = self.indice_abrasion(m_ramales, i)

        return m_ramales
