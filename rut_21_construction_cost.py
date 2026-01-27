from datetime import datetime
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
import re
from natsort import natsorted, ns
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
from scipy.spatial import distance
from shapely.geometry import LineString
from operator import itemgetter
from pathlib import Path
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Add local modules to path
import config
config.setup_sys_path()

from RUT_1 import varGlobals
vg = varGlobals()

from pypiper_compiled import (nb_insert_points_in_linestring)

# --------------------------------------------------------------------------------------------------------------------------------------------
class ExcelFormatter:
    def __init__(self, data, tipo, fase, parameters_dict):
        self.parameters_dict = parameters_dict
        self.tipo = tipo
        self.fase = fase
        self.data = data
        self.df = None
        self.budget_df = None
    
    def flatten_data(self):
        flattened_data = []
        item_counter = 1
        for category, details in self.data.items():
            flattened_data.append((str(item_counter), "", category, "", ""))
            sub_item_counter = 1
            for code, (description, unit) in details.items():
                flattened_data.append((f"{item_counter}.{sub_item_counter}", code, description, unit, "", category))
                sub_item_counter += 1
            item_counter += 1
        self.df = pd.DataFrame(flattened_data, columns=["ITEM", "CODIGO", "RUBRO", "UNIDAD", "CANTIDAD", 'SECTION'])
    
    def clean_and_renumber_dataframe(self, df):
        # Step 1: Identify sections and headers
        df['SECTION'] = df['ITEM'].astype(str).str.split('.').str[0]
        df['IS_HEADER'] = df['ITEM'].astype(str).str.match(r'^\d+$')
        
        # Step 2: Convert 'CANTIDAD' to numeric, treating empty strings as NaN
        df['CANTIDAD'] = pd.to_numeric(df['CANTIDAD'].replace('', np.nan), errors='coerce')
        
        # Step 3: Remove sections where all 'CANTIDAD' values are NaN or zero
        sections_to_keep = np.where(df['CANTIDAD'].fillna(0) > 0, df['CANTIDAD'].fillna(0), 0) + df['IS_HEADER']
        sections_to_keep = np.where(sections_to_keep > 0, True, False)
        df = df[sections_to_keep | df['IS_HEADER']]
        
        # Step 4: Remove headers of completely empty or zero sections
        headers_to_keep = df.groupby('SECTION')['CANTIDAD'].transform(lambda x: (x.notna() & (x > 0)).any())
        df = df[headers_to_keep | ~df['IS_HEADER']]
        
        # Step 5: Renumber the items
        new_items = []
        main_item = 0
        sub_item = 0
        
        for _, row in df.iterrows():
            if row['IS_HEADER']:
                main_item += 1
                sub_item = 0
                new_items.append(str(main_item))
            else:
                sub_item += 1
                new_items.append(f"{main_item}.{sub_item}")
        
        df['ITEM'] = new_items
        
        # Reset index and drop unnecessary columns
        df = df.reset_index(drop=True)
        df = df.drop(['SECTION', 'IS_HEADER'], axis=1)
        df['CANTIDAD'] = df['CANTIDAD'].fillna('')
        
        return df
    
    def remove_trailing_zero(self, serie):
        def clean(x):
            if isinstance(x, (float, int)):
                return f'{x:g}'
            elif isinstance(x, str) and x.endswith('.0'):
                return x[:-2]
            return x
        
        return serie.apply(clean)
    
    def add_budget_values(self):
        # Ensure SECTION names match by trimming whitespace and converting to uppercase
        self.df['SECTION'] = self.df['SECTION'].str.strip().str.upper()
        self.budget_df['SECTION'] = self.budget_df['SECTION'].str.strip().str.upper()
        
        # Perform the merge operation using both 'SECTION' and 'CODIGO'
        self.df_presupuesto = self.df.merge(self.budget_df[['SECTION', 'CODIGO', 'CANTIDAD']], on=['SECTION', 'CODIGO'], how='left', suffixes=('', '_budget'))
        
        # Update CANTIDAD
        aumento_de_cantidades = 1 + self.parameters_dict['aumento_de_cantidades']
        
        # Identify integers and non-NaN values
        mask = self.df_presupuesto['CANTIDAD_budget'].notna()
        is_int = self.df_presupuesto.loc[mask, 'CANTIDAD_budget'].mod(1).eq(0)
        
        # Create new values preserving types
        self.df_presupuesto.loc[mask, 'CANTIDAD_budget'] = np.where(is_int, np.round(self.df_presupuesto.loc[mask, 'CANTIDAD_budget'] * aumento_de_cantidades).astype(int), np.round(self.df_presupuesto.loc[mask, 'CANTIDAD_budget'] * aumento_de_cantidades, 3))
        self.df_presupuesto['CANTIDAD'] = self.df_presupuesto['CANTIDAD_budget'].fillna(self.df_presupuesto['CANTIDAD'])
        
        # Drop the temporary 'CANTIDAD_budget' column
        self.df_presupuesto = self.df_presupuesto.drop('CANTIDAD_budget', axis=1)
        
        # Drop the SECTION column as it's no longer needed after merging
        self.df_presupuesto = self.df_presupuesto.drop('SECTION', axis=1)
        
        self.df_presupuesto = self.clean_and_renumber_dataframe(self.df_presupuesto)
        
        self.df_presupuesto['CODIGO'] = self.df_presupuesto['CODIGO'].astype(str)
        
        # add prefix to titles
        mask = pd.to_numeric(self.df_presupuesto['ITEM'], errors='coerce').apply(lambda x: x.is_integer() if pd.notnull(x) else False)
        
        # definir codigo
        if "pluvial" in self.tipo.lower():
            prefix_string = "alc-pluvial"
        elif "sanitario" in self.tipo.lower():
            prefix_string = "alc-sanitario"
        elif "combinado" in self.tipo.lower():
            prefix_string = "alc-combinado"
        else:
            prefix_string = "alc"
            
        prefix_string = ' - '.join([prefix_string.upper(), ('componente ' + self.fase).upper(), ''])
        # Add prefix only to the specified indices
        self.df_presupuesto['RUBRO'][mask] = prefix_string + self.df_presupuesto['RUBRO'].loc[mask]
        
        # -------------------------------------------------------------------------------------------------------------------------------------------
        base_precios = self.parameters_dict['base_precios']

        
        df_base_precio = pd.read_excel(base_precios, dtype={'codigo_alterno': str, 'codigo': str, 'precio': float, 'descripccion': str, 'unidad': str})
        codigo_map_df = df_base_precio.copy()

        # codigo_map_df = codigo_map_df[codigo_map_df['codigo_alterno'].notna() & (codigo_map_df['codigo_alterno'].str.strip() != '')]
        
        codigo_map_df = codigo_map_df.dropna(subset=['codigo_alterno'])
        # codigo_map_df['codigo'] = codigo_map_df['codigo'].astype(str).str.strip()
        # codigo_map_df['codigo_alterno'] = codigo_map_df['codigo_alterno'].astype(str).str.strip()
        
        # -----change codigo EPMAPS to codigo INTERPRO
        a_dict = dict(zip(codigo_map_df['codigo_alterno'], codigo_map_df['codigo']))
        b = self.df_presupuesto['CODIGO'].map(a_dict).dropna()
        self.df_presupuesto['CODIGO'].loc[b.index] = b
        
        # -----change items names from base file
        c_dict = dict(zip(codigo_map_df['codigo'], codigo_map_df['descripccion']))
        d = self.df_presupuesto['CODIGO'].map(c_dict).dropna()
        self.df_presupuesto['RUBRO'].loc[d.index] = d
        
        # get cost
        codigo_map_df = df_base_precio.copy()
        
        # codigo_map_df['codigo'] = self.remove_trailing_zero(codigo_map_df['codigo']).astype(str).str.strip()
        # codigo_map_df['codigo_alterno'] = codigo_map_df['codigo_alterno'].astype(str).str.strip()
        precio_dict = dict(zip(codigo_map_df['codigo'], codigo_map_df['precio']))
        # self.df_presupuesto['PRECIO UNITARIO'] = self.remove_trailing_zero(self.df_presupuesto['CODIGO']).astype(str).map(precio_dict)
        self.df_presupuesto['PRECIO UNITARIO'] = self.df_presupuesto['CODIGO'].map(precio_dict)
        self.df_presupuesto['PRECIO TOTAL'] = pd.to_numeric(self.df_presupuesto['PRECIO UNITARIO']) * pd.to_numeric(self.df_presupuesto['CANTIDAD'])
        
        self.total_budget = self.df_presupuesto['PRECIO TOTAL'].sum()
        self.df_presupuesto['grupo'] = self.df_presupuesto['ITEM'].str.split('.', expand=True).iloc[:, 0]
        for _, grupo in self.df_presupuesto.groupby('grupo'):
            self.df_presupuesto.loc[grupo.index[0], 'PRECIO TOTAL'] = grupo['PRECIO TOTAL'].sum()
        
        self.df_presupuesto = self.df_presupuesto.drop('grupo', axis=1)
        self.df_presupuesto = self.df_presupuesto.fillna('')
    
    def match_and_get_codes(self, input_df, rubros_dict):
        result = {'CODIGO': [], 'CANTIDAD': [], 'SECTION': []}
        
        for idx, row in input_df.iterrows():
            section = row['SECTION']
            value = row['CANTIDAD']
            try:
                matched_key = self.find_best_match(idx, rubros_dict[section])
            except:
                print('error', idx, section)
            if not matched_key:
                print(idx, section)
            
            if matched_key:
                if not idx == matched_key:
                    # print(idx, matched_key, section)
                    pass
                code = rubros_dict[section][matched_key]
                if code != '*':
                    result['CODIGO'].append(code)
                    result['CANTIDAD'].append(value)
                    result['SECTION'].append(section)
        
        return pd.DataFrame(result)
    
    def find_best_match(self, idx, rubros_dict):
        # Normalize both the input string and dictionary keys
        simplified_idx = re.sub(r'(\d+)\.0', r'\1', idx)
        simplified_keys = {re.sub(r'(\d+)\.0', r'\1', k): k for k in rubros_dict.keys()}
        
        # Use process.extractOne to find the best match
        best_match = process.extractOne(simplified_idx, simplified_keys.keys(), scorer=fuzz.ratio)
        
        if best_match and best_match[1] > 80:  # Higher threshold, but using simplified keys
            return simplified_keys[best_match[0]]
        return None
    
    def create_budget_data(self, rubros_dict, results_df):
        """
        Create budget data based on a rubros dictionary and results DataFrame.
    
        Args:
        rubros_dict (dict): Dictionary with excavation categories as keys and codes as values.
        results_df (pd.Series): Series with results, with index matching some keys in rubros_dict.
    
        Returns:
        pd.DataFrame: A DataFrame with 'CODIGO' and 'CANTIDAD' columns for budget data.
        """
        
        self.budget_df = self.match_and_get_codes(results_df, rubros_dict)
    
    def create_header(self, worksheet, workbook, proyecto_name, sistema, ubicacion, obra, cliente, fecha, presupuesto):
        # Define formats
        title_format = workbook.add_format({'font_name': 'Arial', 'font_size': 12, 'bold': True, 'align': 'center', 'border': 1})
        header_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'bold': True, 'align': 'right'})
        value_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'align': 'left'})
        budget_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'bold': True, 'align': 'left', 'num_format': '$#,##0.00'})
        border_format = workbook.add_format({'border': 1})
        # Write project name with border
        worksheet.merge_range('A1:G1', proyecto_name, title_format)
        # Write other header information
        headers = [('A2:B2', 'SISTEMA:', header_format), ('C2', sistema, value_format), ('D2:E2', 'CLIENTE:', header_format), ('F2:G2', cliente, value_format), ('A3:B3', 'UBICACIÓN :', header_format), ('C3', ubicacion, value_format), ('D3:E3', 'FECHA:', header_format), ('F3:G3', fecha, value_format), ('A4:B4', 'OBRA:', header_format), ('C4', obra, value_format), ('D4:E4', 'PRESUPUESTO', header_format), ('F4:G4', presupuesto, budget_format)]
        for cell, value, cell_format in headers:
            if ':' in cell:
                worksheet.merge_range(cell, value, cell_format)
            else:
                worksheet.write(cell, value, cell_format)
        
        # Add only outer border to the header section
        outer_border_format = workbook.add_format({'left': 1, 'right': 1, 'top': 1, 'bottom': 1})
        worksheet.conditional_format('A1:G1', {'type': 'formula', 'criteria': 'TRUE', 'format': outer_border_format})
        
        # Add outer border around the header section
        border_format = workbook.add_format({'left': 0, 'right': 1, 'top': 0, 'bottom': 0})
        worksheet.conditional_format('G1:G5', {'type': 'formula', 'criteria': 'TRUE', 'format': border_format})
        
        # Add outer border around the header section
        border_format = workbook.add_format({'left': 0, 'right': 1, 'top': 1, 'bottom': 1})
        worksheet.conditional_format('G6', {'type': 'formula', 'criteria': 'TRUE', 'format': border_format})
    
    def to_excel(self, filename, proyecto_name, sistema, ubicacion, obra, cliente):
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        
        # Write the dataframe to the worksheet
        self.df_presupuesto.to_excel(writer, sheet_name='Presupuesto', index=False, startrow=5)  # Start from row 6
        
        # Get the workbook and the worksheet
        workbook = writer.book
        worksheet = writer.sheets['Presupuesto']
        
        fecha = datetime.now().strftime('%d-%b-%Y')
        presupuesto = self.total_budget
        
        # Create header
        self.create_header(worksheet, workbook, proyecto_name, sistema, ubicacion, obra, cliente, fecha, presupuesto * 1.15)
        
        # Define formats for the main data
        regular_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'border': 1})
        bold_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'bold': True, 'border': 1})
        title_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'bold': True, 'border': 1, 'bg_color': '#D9D9D9'})
        money_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'border': 1, 'num_format': '$#,##0.00'})
        money_bold_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'bold': True, 'border': 1, 'num_format': '$#,##0.00'})
        money_title_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'bold': True, 'border': 1, 'bg_color': '#D9D9D9', 'num_format': '$#,##0.00'})
        
        # Specify which columns should have money format
        money_columns = ['PRECIO UNITARIO', 'PRECIO TOTAL']  # Add or remove column names as needed
        
        # Get the indices of money columns
        money_col_indices = [self.df_presupuesto.columns.get_loc(col) for col in money_columns if col in self.df_presupuesto.columns]
        
        # Apply formatting to the data
        codigo_col = self.df_presupuesto.columns.get_loc('CODIGO')
        item_col = self.df_presupuesto.columns.get_loc('ITEM')
        for row in range(self.df_presupuesto.shape[0] + 1):  # +1 to include the header row
            is_title = (row > 0 and row < self.df_presupuesto.shape[0] and self.df_presupuesto.iloc[row - 1, codigo_col] == "" and self.df_presupuesto.iloc[row - 1, item_col] != "")
            for col in range(self.df_presupuesto.shape[1]):
                value = self.df_presupuesto.columns[col] if row == 0 else self.df_presupuesto.iloc[row - 1, col]
                if col in money_col_indices:
                    if row == 0:
                        worksheet.write(row + 5, col, value, money_bold_format)
                    elif is_title:
                        worksheet.write(row + 5, col, value, money_title_format)
                    else:
                        worksheet.write(row + 5, col, value, money_format)
                elif row == 0:
                    worksheet.write(row + 5, col, value, bold_format)
                elif is_title:
                    worksheet.write(row + 5, col, value, title_format)
                else:
                    worksheet.write(row + 5, col, value, regular_format)
        
        # Adjust column widths
        for col in range(self.df_presupuesto.shape[1]):
            max_length = max(self.df_presupuesto.iloc[:, col].astype(str).map(len).max(), len(self.df_presupuesto.columns[col]))
            worksheet.set_column(col, col, max_length + 2)
        
        # ---------------------------------------------------------------
        # Find the last row and the 'PRECIO TOTAL' column
        last_row = self.df_presupuesto.shape[0] + 6  # +6 because we started data at row 6
        precio_total_col = self.df_presupuesto.columns.get_loc('PRECIO TOTAL')
        # Define new formats
        bold_left_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'bold': True, 'align': 'left', 'border': 1})
        bold_right_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'bold': True, 'align': 'right', 'border': 1})
        bold_money_format = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'bold': True, 'align': 'right', 'num_format': '$#,##0.00', 'border': 1})
        # Add one empty row
        last_row += 1
        # Calculate values
        total = self.total_budget
        iva = total * 0.15
        total_presupuesto = total + iva
        # Write the final section
        start_row = last_row
        worksheet.merge_range(last_row, precio_total_col - 3, last_row, precio_total_col - 1, 'TOTAL SIN IVA', bold_left_format)
        worksheet.write(last_row, precio_total_col, total, bold_money_format)
        last_row += 1
        worksheet.merge_range(last_row, precio_total_col - 3, last_row, precio_total_col - 2, 'IMPUESTO IVA', bold_left_format)
        worksheet.write(last_row, precio_total_col - 1, '15.00%', bold_right_format)
        worksheet.write(last_row, precio_total_col, iva, bold_money_format)
        last_row += 1
        worksheet.merge_range(last_row, precio_total_col - 3, last_row, precio_total_col - 1, 'TOTAL DEL PRESUPUESTO', bold_left_format)
        worksheet.write(last_row, precio_total_col, total_presupuesto, bold_money_format)
        # Add outer border to the entire resumen section
        worksheet.conditional_format(start_row, precio_total_col - 3, last_row, precio_total_col, {'type': 'formula', 'criteria': 'True', 'format': workbook.add_format({'border': 1})  # Changed from 2 to 1 for normal border
                                                                                                   })
        # Close the Pandas Excel writer and output the Excel file
        writer.close()


class TrenchWidthCalculator:
    def __init__(self, ):
        # Define the rules for Sobre Ancho
        self.columns = ['De 0 m a 2.75 m s/entibado', 'De 0 m a 2.75 m c/entibado', 'De 2.75 m a 4 m s/entibado', 'De 2.75 m a 4 m c/entibado', 'De 4 m a 6 m s/entibado', 'De 4 m a 6 m c/entibado', 'De 6 m a inf m s/entibado', 'De 6 m a inf m c/entibado']
        self.sa_rules = {(0, 2.75): 0.15, (2.75, 4): 0.20, (4, 6): 0.30, (6, float('inf')): 0.40}
    
    def extract_diameter_range(self, diametro_interno_externo_pypiper):
        """
        Extracts all unique outer diameters, converts them from meters to millimeters,
        rounds them to the nearest 100 mm, and filters them to be between 100 mm and 2500 mm.
        """
        all_outer_diameters = []
        for material, sizes in diametro_interno_externo_pypiper.items():
            if isinstance(sizes, dict):
                all_outer_diameters.extend(sizes.values())
            else:
                all_outer_diameters.extend(list(sizes))
        
        # Convert from meters to mm, include up to 2.5 meters, and round to nearest 100 mm
        valid_diameters = np.unique(np.round([d * 1000 for d in all_outer_diameters], -2))
        return valid_diameters
    
    def extract_range_depth(self, text):
        """Extract depth range from column name, including infinity."""
        # Pattern for regular range
        pattern = r'De (\d+(?:\.\d+)?) m a (\d+(?:\.\d+)?) m'
        matches = re.findall(pattern, text)
        
        if matches:
            return [float(matches[0][0]), float(matches[0][1])]
        
        # Pattern for infinity range
        inf_pattern = r'De (\d+(?:\.\d+)?) m a inf m'
        inf_matches = re.findall(inf_pattern, text)
        
        if inf_matches:
            return [float(inf_matches[0]), np.inf]
        
        # If no match found, return None or raise an exception
        return None  # or raise ValueError(f"Unable to extract range from: {text}")
    
    def generate_diameter_table(self, diametro_interno_externo_pypiper):
        diameters = self.extract_diameter_range(diametro_interno_externo_pypiper)
        # Create a DataFrame with diameters as index and specified columns
        df = pd.DataFrame(index=diameters, columns=self.columns)
        
        # Fill the DataFrame
        for col in self.columns:
            start, end = self.extract_range_depth(col)
            sa = self.get_sa((start + end) / 2)  # Use end depth to determine SA
            
            for diameter in diameters:
                base_width = diameter / 1000 + 0.5  # Convert mm to m and add 0.5
                if 's/entibado' in col:
                    df.loc[diameter, col] = round(base_width, 2)  # No shoring: diameter/1000 + 0.5
                else:  # c/entibado
                    df.loc[diameter, col] = round(base_width + sa, 2)  # With shoring: diameter/1000 + 0.5 + SA
        
        df['Diametro'] = diameters
        return df
    
    def get_sa(self, depth):
        
        for (start, end), value in self.sa_rules.items():
            if start <= depth < end:
                return value
        return list(self.sa_rules.values())[-1]  # Default value for depths > 6m)
    
    def round_and_convert(self, df):
        # Round to nearest 0.05
        df_rounded = np.round(df / 0.05) * 0.05
        # Convert to dictionary of lists
        trench_dict = df_rounded.to_dict(orient='list')
        return trench_dict


class CantidadesZanjaAbierta:
    def __init__(self, parameters_dict):
        # Initialize tables and parameters
        self.df_ancho = pd.DataFrame(parameters_dict['tabla_ancho']).set_index('Diametro')
        self.df_taludes = pd.DataFrame(parameters_dict['tabla_taludes']).set_index('h (m)')
        self.entibado = {True: 'c/entibado', False: 's/entibado'}
        
        # parameters for shoring type determination
        self.altura_minima_entibado_discontinuo_madera = parameters_dict['altura_minima_entibado_discontinuo_madera']
        self.altura_minima_entibado_continuo_madera = parameters_dict['altura_minima_entibado_continuo_madera']
        self.altura_minima_entibado_continuo_metalico = parameters_dict['altura_minima_entibado_continuo_metalico']
        
        # Convert DataFrame to NumPy array for faster indexing
        self.ancho_array = self.df_ancho.to_numpy()
        self.diameter_values = self.df_ancho.index.to_numpy()
        self.column_names = self.df_ancho.columns.to_numpy()
        
        # Extract depth ranges from column names
        self.depth_ranges = np.array([self.extract_range_depth(col) for col in self.column_names if col.startswith('De')])
        
        # Prepare taludes data for vectorized operations
        self.depth_thresholds = self.df_taludes.index.to_numpy()
        self.slope_values = self.df_taludes['1/x'].to_numpy()
        
        # class percentage of soils
        self.tabla_tipo_suelo = parameters_dict['tabla_tipo_suelo']
        self.tabla_tipo_suelo_maquina = parameters_dict['tabla_tipo_suelo']['tipo_suelo_maquina']
        self.tabla_tipo_suelo_mano = parameters_dict['tabla_tipo_suelo']['tipo_suelo_mano']
        
        self.porcentage_desbroce = parameters_dict['porcentage_desbroce']
        self.distancia_acarreo_manual = parameters_dict['distancia_acarreo_manual']
        self.distancia_desalojo = parameters_dict['distancia_desalojo']
        self.porcentage_esponjamiento = parameters_dict['porcentaje_esponjamiento']
        
        # get vector path
        self.vector_path = parameters_dict['vector_path']
        try:
            self.m_ramales_df = gpd.read_file(self.vector_path, engine='pyogrio')
        except:
            self.m_ramales_df = gpd.read_file(self.vector_path)
        
        filtro_dict = parameters_dict.get('filtro')
        if filtro_dict:
            filtro_column = filtro_dict.get('column')
            filtro_value = filtro_dict.get('value')
            
            if filtro_column and filtro_value:
                # Convert filtro_value to a list if it's a string of comma-separated values
                if isinstance(filtro_value, str):
                    filtro_value = [x.strip() for x in filtro_value.split(',')]
                
                # Apply the filter using isin to handle lists of values
                filtro = self.m_ramales_df[filtro_column].isin(filtro_value)
                self.m_ramales_df = self.m_ramales_df.loc[filtro]
        
        # volumen de tramos nuevos
        filtro_nuevo = self.m_ramales_df['Estado'] == 'nuevo'
        self.m_ramales_df = self.m_ramales_df.loc[filtro_nuevo]
        
        self.longitud_total = self.m_ramales_df['L'].sum()
        
        # tramos solo  zanja
        self.metodos_constructivos_maquina = ['zanja abierta', 'colector']
        self.metodos_constructivos_mano = ['zanja mano']
        filtro_metodo_constructivo = self.m_ramales_df['metodo_constructivo'].str.contains("|".join(self.metodos_constructivos_mano + self.metodos_constructivos_maquina), case=False, na=False)
        self.m_ramales_df = self.m_ramales_df.loc[filtro_metodo_constructivo]
        
        self.m_ramales_df = self.m_ramales_df.reset_index(drop=True)

        # if self.m_ramales_df.empty:
        #     sys.exit(f"ADVERTENCIA: el filtro resulto en un dataframe vacio, por favor revisa que existan tramos nuevos y al menos una seccion sea circular.")
    
    def extract_range_depth_old(self, text):
        """Extract depth range from column name."""
        pattern = r'De (\d+(?:\.\d+)?) m a (\d+(?:\.\d+)?) m'
        matches = re.findall(pattern, text)
        return [float(_) for _ in matches[0]]
    
    def extract_range_depth(self, text):
        """Extract depth range from column name, including infinity."""
        # Pattern for regular range
        pattern = r'De (\d+(?:\.\d+)?) m a (\d+(?:\.\d+)?) m'
        matches = re.findall(pattern, text)
        
        if matches:
            return [float(matches[0][0]), float(matches[0][1])]
        
        # Pattern for infinity range
        inf_pattern = r'De (\d+(?:\.\d+)?) m a inf m'
        inf_matches = re.findall(inf_pattern, text)
        
        if inf_matches:
            return [float(inf_matches[0]), np.inf]
        
        # If no match found, return None or raise an exception
        return None  # or raise ValueError(f"Unable to extract range from: {text}")
    
    def get_depth_range_index(self, depth):
        """Get the index of the depth range for a given depth."""
        for i, (start, end) in enumerate(self.depth_ranges):
            if start <= depth < end:
                return i
        return len(self.depth_ranges) - 1  # Return the last index if depth exceeds all ranges
    
    def determine_shoring_type(self, depth):
        if depth < self.altura_minima_entibado_discontinuo_madera:
            return False  # No shoring needed
        elif depth < self.altura_minima_entibado_continuo_madera:
            return True  # Discontinuous shoring
        elif depth < self.altura_minima_entibado_continuo_metalico:
            return True  # Discontinuous shoring
        else:
            return True  # Continuous shoring (we'll treat this the same as discontinuous for our purposes)
    
    def normalize_string(self, s):
        # Remove whitespace and convert to lowercase
        s = ''.join(s.split()).lower()
        # Replace decimal numbers with a standardized format
        s = re.sub(r'(\d+)\.0+', r'\1', s)
        return s
    
    def find_matching_indices(self, list1, list2):
        normalized_list1 = [self.normalize_string(s) for s in list1]
        
        matching_indices = []
        
        for item in list2:
            normalized_item = self.normalize_string(item)
            best_match = process.extractOne(normalized_item, normalized_list1, scorer=fuzz.partial_ratio)
            
            if best_match and best_match[1] > 80:  # 80 is the threshold, adjust as needed
                match_idx = normalized_list1.index(best_match[0])
                matching_indices.append(match_idx)
            else:
                matching_indices.append(None)  # No good match found
        
        return matching_indices
    
    # -------------------------------------------------------------------------------------------------------------------
    def get_ancho(self, diameters, depths):
        """
        Calculate the base width of the trench for given diameters, shorings, and depths.
        
        Args:
        diameters (np.array): Array of pipe diameters in mm.
        shorings (np.array): Array of boolean values indicating shoring status.
        depths (np.array): Array of trench depths in meters.
        
        Returns:
        np.array: Array of calculated base widths for the trench.
        """
        
        # Find nearest diameters
        diameter_indices = np.abs(self.diameter_values[:, np.newaxis] / 1000 - diameters).argmin(axis=0)
        
        # Determine shoring type
        shorings = np.array([self.determine_shoring_type(d) for d in depths])
        self.shoring_array = shorings
        
        # Determine depth range indices
        depth_indices = np.array([self.get_depth_range_index(d) for d in depths])
        
        # Generate column names
        selected_depth_ranges = self.depth_ranges[depth_indices]
        entibado_status = np.array([self.entibado[s] for s in shorings])
        columns = [f"De {start} m a {end} m {status}" for (start, end), status in zip(selected_depth_ranges, entibado_status)]
        
        column_indices = np.array(self.find_matching_indices(self.df_ancho.columns.to_list(), columns))
        
        # Use NumPy fancy indexing to get ancho values
        ancho_values = self.ancho_array[diameter_indices, column_indices]
        
        return ancho_values
    
    def get_trench_slope(self, depths):
        """
        Assign slopes based on depth thresholds.
        
        Args:
        depths (np.array): Array of depths to assign slopes to.
        
        Returns:
        np.array: Array of assigned slopes.
        """
        # Use searchsorted to find the indices of the appropriate slopes
        indices = np.searchsorted(self.depth_thresholds, depths, side='right')
        
        # Assign slopes based on the indices
        return self.slope_values[np.minimum(indices, len(self.slope_values) - 1)]
    
    def get_trapezoid_area(self, base_width, depth, slope):
        """
        Calculate the area of the trapezoidal cross-section of the trench.
        
        Args:
        base_width (np.array): Width of the base of the trench (bm in the diagram).
        depth (np.array): Depth of the trench (hm in the diagram).
        slope (np.array): Slope of the trench sides (1/x value).
        
        Returns:
        np.array: Array of areas of the trapezoids.
        """
        
        # Calculate the top width of the trapezoid
        top_width = base_width + 2 * depth / slope
        
        # Calculate the area using the trapezoid formula
        area = (base_width + top_width) * depth / 2
        
        return area
    
    def calculate_trench_areas(self, depth_ranges, base_widths, trench_depths, slopes):
        """
        Calculate trench areas for different depth ranges.
        
        Args:
        depth_ranges (list of lists): List of depth ranges, each as [start, end].
        base_widths (np.array): Array of base widths for each trench.
        trench_depths (np.array): Array of total depths for each trench.
        slopes (np.array): Array of slope values for each trench (1/x values).
        
        Returns:
        pd.DataFrame: DataFrame with areas for each depth range and trench.
        """
        results = {}
        for pos, depth_range in enumerate(depth_ranges):
            empty_arr = np.empty(len(trench_depths))
            
            start, end = depth_range
            condition_in = np.logical_and(trench_depths >= start, trench_depths < end)
            condition_out = np.invert(condition_in)
            
            if start == 0:
                empty_arr[condition_in.nonzero()] = self.get_trapezoid_area(base_widths[condition_in], trench_depths[condition_in], slopes[condition_in])
                empty_arr[condition_out.nonzero()] = self.get_trapezoid_area(base_widths[condition_out], end - start, slopes[condition_out])
                results[f'{start}m-{end}m'] = empty_arr
            else:
                previus_filtro = np.invert(trench_depths <= start)
                new_base = (start / slopes) * 2 + base_widths
                empty_arr[condition_in.nonzero()] = self.get_trapezoid_area(new_base[condition_in], trench_depths[condition_in] - start, slopes[condition_in])
                empty_arr[condition_out.nonzero()] = self.get_trapezoid_area(new_base[condition_out], end - start, slopes[condition_out]) * previus_filtro[condition_out]
                results[f'{start}m-{end}m'] = empty_arr
        
        # Create DataFrame with labeled columns for each depth range
        df = pd.DataFrame(results)
        return df
    
    def get_range_areas(self, diameters: np.array, depths: np.array) -> pd.DataFrame:
        """
       Calculate the area for each depth range based on given diameters and depths.
    
       Args:
       diameters (np.array): Array of pipe diameters in mm.
       depths (np.array): Array of trench depths in meters.
    
       Returns:
       pd.DataFrame: DataFrame with columns for each depth range and rows for each input.
       """
        
        # Calculate base widths and slopes for all inputs at once
        base_widths = self.get_ancho(diameters, depths)
        slopes = self.get_trench_slope(depths)
        
        # get areas by depth
        df = self.calculate_trench_areas(np.unique(self.depth_ranges, axis=0), base_widths, depths, slopes)
        
        df.fillna(0, inplace=True)
        
        # volumenes por tramo
        self.vol_rows = df.copy().mul(self.m_ramales_df['L'].to_numpy(), axis=0)
        self.area_properties = pd.DataFrame(self.m_ramales_df.index)
        self.area_properties['tramo'] = self.m_ramales_df['Tramo'].to_numpy()
        self.area_properties['metodo_constructivo'] = self.m_ramales_df['metodo_constructivo'].to_numpy()
        self.area_properties['diametro'] = self.seccion_str2float(self.m_ramales_df['D_ext'].to_numpy(), return_b=True)
        self.area_properties['profundidad'] = np.mean([self.m_ramales_df['HF'].to_numpy(), self.m_ramales_df['HI'].to_numpy()], axis=0)
        self.area_properties['longitud'] = self.m_ramales_df['L'].to_numpy()
        self.area_properties['base_inferior'] = base_widths
        self.area_properties['base_superior'] = (self.area_properties['profundidad'] / slopes) * 2 + base_widths
        self.area_properties['talud'] = slopes
        
        return df
    
    def check_totals(self, tabla_tipo_suelo):
        for categoria, valores in tabla_tipo_suelo.items():
            total = sum(valores.values())
            # Imprime la categoría y la suma total de sus valores
            print(f"Total para {categoria}: {total * 100}%")
            if total != 1:
                print(f"Advertencia: El total para {categoria} no suma 100%.")
    
    def seccion_str2float(self, arr, return_b=False):
        """
        :param arr: array of string mix of circular and rectangular sections
        :return: array of diameter of circular sections and height dimensions of rectangular sections
        """
        # Convert input to pandas Series if it's not already
        arr_series = pd.Series(arr)
        
        # Check if there are any 'x' in the array
        if arr_series.str.contains('x').any():
            # Split base and height when possible
            split_arr = arr_series.str.split('x', expand=True)
            b = split_arr[0].fillna(0).to_numpy().astype(str)
            h = split_arr[1].fillna(0).to_numpy().astype(str)
            
            # Get circular and rectangular indexes
            circular_index = np.char.equal(h, '0').nonzero()[0]
            rectangular_index = np.char.not_equal(h, '0').nonzero()[0]
            
            # Zeros array
            out = np.zeros(shape=len(b), dtype=float)
            
            # Fill zeros array
            out[circular_index] = b[circular_index].astype(float)
            
            if return_b:
                out[rectangular_index] = b[rectangular_index].astype(float)
            else:
                out[rectangular_index] = h[rectangular_index].astype(float)
        else:
            # If there are no 'x', treat all as circular sections
            out = arr_series.astype(float).to_numpy()
        
        return out
    
    def get_depth_vol_quantities(self):
        
        # get diamenters
        diameters = self.seccion_str2float(self.m_ramales_df['D_ext'], return_b=True)
        
        # get diamenters
        depths = np.round(np.mean([self.m_ramales_df['HF'], self.m_ramales_df['HI']], axis=0), 4)
        
        # get areas
        range_areas = self.get_range_areas(diameters, depths)
        
        vol = np.round(range_areas.to_numpy() * self.m_ramales_df['L'].to_numpy()[:, np.newaxis], 1)
        vol_df = pd.DataFrame(vol, columns=range_areas.columns)
        
        return vol_df
    
    def process_vol_rows(self, df, soil_types):
        depth_ranges = df.columns.to_list()
        # Initialize a list to store results
        results = []
        # Process each depth range
        for depth in depth_ranges:
            if depth in df.columns:
                total_volume = df[depth].sum()
                for soil_type, percentage in soil_types.items():
                    index = f"{depth}-{soil_type}"
                    volume = total_volume * percentage
                    results.append((index, volume))
        # Convert results to Series
        result_series = pd.Series(dict(results))
        return result_series
    
    def get_class_soil_vol_quantities(self, force_general_percentages=False):
        
        # get volumne based on the depth ranges
        vol = self.get_depth_vol_quantities()
        
        # verificar si en los metodos constructivos hay zanja mano, si no hay, se aplica los porcentages estandar
        condicion_zanja_mano = self.m_ramales_df['metodo_constructivo'].str.contains("|".join(self.metodos_constructivos_mano), case=False, na=False).to_numpy()
        condicion_zanja_maquina = self.m_ramales_df['metodo_constructivo'].str.contains("|".join(self.metodos_constructivos_maquina), case=False, na=False).to_numpy()
        
        if len(condicion_zanja_mano.nonzero()[0]) == 0 or force_general_percentages:
            new_column_names = []
            new_column_values = []
            for col in vol.columns:
                s = vol[col]
                for soil_class, soil_percentage in self.tabla_tipo_suelo['tipo_suelo'].items():
                    new_column_names.append(col + '-' + soil_class)
                    new_column_values.append(s.to_numpy() * soil_percentage)
            
            # get the value for each class of soil
            vol_soil_class = pd.DataFrame(np.array(new_column_values).T, columns=new_column_names)
            vol_soil_class_sum = vol_soil_class.sum(axis=0)
            vol_soil_class_sum = vol_soil_class_sum.to_frame()
            map_tipo_excavacion = {'manual': 'EXCAVACION A MANO', 'mecanica': 'EXCAVACION MECANICA'}
            seccion = vol_soil_class_sum.index.map(lambda x: next((v for k, v in map_tipo_excavacion.items() if k in x), None)).to_list()
            vol_soil_class_sum.columns = ['CANTIDAD']
            vol_soil_class_sum['SECTION'] = seccion
        
        else:
            cantidades_mano = self.process_vol_rows(self.vol_rows[condicion_zanja_mano], self.tabla_tipo_suelo_mano)
            cantidades_mano = cantidades_mano.to_frame()
            cantidades_mano.columns = ['CANTIDAD']
            cantidades_mano['SECTION'] = 'EXCAVACION A MANO'
            
            cantidades_maquina = self.process_vol_rows(self.vol_rows[condicion_zanja_maquina], self.tabla_tipo_suelo_maquina)
            cantidades_maquina = cantidades_maquina.to_frame()
            cantidades_maquina.columns = ['CANTIDAD']
            cantidades_maquina['SECTION'] = 'EXCAVACION MECANICA'
            
            vol_soil_class_sum = pd.concat([cantidades_mano, cantidades_maquina])
        
        return vol_soil_class_sum
    
    # -------------------------------------------------------------------------------------------------------------------
    
    def categorize_entibado(self, row):
        # Suelo firme (N60 > 4)
        if row['N60'] > 6:
            if row["HF"] < 0.5:
                return "Sin entibado"
            elif row["HF"] < 0.9:
                return "Entibado discontinuo madera"
            elif row["HF"] <= 2.0:
                return "Entibado continuo madera"
            else:  # HF > 2.0
                return "Entibado continuo metalico"
        
        # Suelo blando (N60 <= 4)
        else:
            if row["HF"] <= 2.0:
                return "Entibado continuo madera"
            else:  # HF > 2.0
                return "Entibado continuo metalico"
    
    def get_shoring(self):
        """
        Calculate the area of shoring for different types based on trench depth.
        
        Returns:
        pandas.Series: Series with shoring types as index and their corresponding areas as values, excluding 'Sin entibado'
        """
        top_start = self.m_ramales_df['ZTI']
        bottom_start = self.m_ramales_df['ZFI']
        top_end = self.m_ramales_df['ZTF']
        bottom_end = self.m_ramales_df['ZFF']
        length = self.m_ramales_df['L']
        
        # Calculate heights at start and end
        height_start = top_start - bottom_start
        height_end = top_end - bottom_end
        
        # Calculate average height
        avg_height = (height_start + height_end) / 2
        # avg_height = np.max([height_start,  height_end], axis=0)
        
        # Calculate area for both sies of the trench
        area = avg_height * length * 2
        
        # Define thresholds for shoring types
        thresholds = [self.altura_minima_entibado_discontinuo_madera, self.altura_minima_entibado_continuo_madera, self.altura_minima_entibado_continuo_metalico]
        
        # Determine shoring type based on average height
        shoring_indices = np.searchsorted(thresholds, avg_height, side='right')
        
        shoring_types = ['Sin entibado', 'Entibado discontinuo madera', 'Entibado continuo madera', 'Entibado continuo metalico']
        
        if 'N60' in self.m_ramales_df.columns:
            shoring_type = self.m_ramales_df.apply(self.categorize_entibado, axis=1).to_numpy()
        else:
            shoring_type = [shoring_types[i] for i in shoring_indices]
        
        # Create DataFrame with results
        results_df = pd.DataFrame({'Shoring_Type': shoring_type, 'Area': area})
        
        # Exclude 'Sin entibado', group by shoring type, sum the areas, and convert to Series
        grouped_results = (results_df[results_df['Shoring_Type'] != 'Sin entibado'].groupby('Shoring_Type')['Area'].sum())
        
        # asegurar que se ponga un valor minimo en todos los casos de entibado
        grouped_results = grouped_results[grouped_results.index != 'Sin entibado']
        # Replace values < 1 with 100
        grouped_results = grouped_results.mask(grouped_results < 1, 100)
        
        self.df_shoring = grouped_results.copy()
        # do not  make it wihtou .copy()!!!
        grouped_results = grouped_results.to_frame()
        grouped_results.columns = ['CANTIDAD']
        grouped_results['SECTION'] = 'ENTIBADO'
        
        return grouped_results
    
    # -------------------------------------------------------------------------------------------------------------------
    
    def map_road_surface(self, row, surface_dict):
        """
        Map the road surface characteristics to the corresponding row in the DataFrame.
    
        This function takes a row from the DataFrame and a dictionary of surface types,
        and populates the row with the appropriate values based on the surface type.
    
        :param row: A row from the DataFrame, containing at least a 'sup_road' column
        :param surface_dict: A dictionary where keys are surface types and values are
                             dictionaries of layer characteristics
        :return: The updated row with mapped surface characteristics
    
        Example:
        If surface_dict = {
            'pavimento_flexible': {'superficie_base': 0.2, 'superficie_subbase': 0.3}
        }
        and row['sup_road'] = 'pavimento_flexible',
        then the function will set row['superficie_base'] = 0.2 and row['superficie_subbase'] = 0.3
        """
        # Get the surface type from the row
        surface_type = row['sup_road']
        
        # Check if the surface type exists in the provided dictionary
        if surface_type in surface_dict:
            # Iterate through the characteristics of this surface type
            for key, value in surface_dict[surface_type].items():
                # If the characteristic (key) exists as a column in the row,
                # update its value
                if key in row.index:
                    row[key] = value
        
        # Return the updated row
        return row
    
    def adjust_layers_old(self, row, available_depth):
        """
        Adjust the layers of road fill to fit within the available depth.
        
        :param row: A row from the DataFrame containing layer heights
        :param available_depth: The available depth for fill (trench depth minus pipe height)
        :return: Adjusted row with updated layer heights
        """
        # Get all unique layer names from tabla_relleno_suelo
        all_layers = set(['sup_road'])
        for surface_type, layers in vg.tabla_relleno_suelo.items():
            if isinstance(layers, dict):
                all_layers.update(layers.keys())
        
        # Sort layers based on their typical order
        layer_order = sorted(all_layers, key=lambda x: vg.tipos_relleno.index(x) if x in vg.tipos_relleno else float('inf'))
        
        # Remove 'sup_road' from layer_order as it's not a layer height
        layer_order.remove('sup_road')
        
        # Calculate the sum of all layers except fondo_mejoramiento
        upper_layers_sum = sum(row[layer] for layer in layer_order if layer != 'fondo_mejoramiento' and pd.notnull(row[layer]))
        
        # Calculate the remaining depth for fondo_mejoramiento
        remaining_depth = available_depth - upper_layers_sum
        
        # Adjust fondo_mejoramiento based on the relleno_mejoramiento percentage
        if remaining_depth > 0 and row['sup_road'] not in ['pasto']:
            row['fondo_mejoramiento'] = remaining_depth * vg.tabla_relleno_suelo['relleno_mejoramiento']
        elif remaining_depth > 0 and row['sup_road'] in ['pasto']:
            row['fondo_mejoramiento'] = vg.tabla_relleno_suelo[row['sup_road']]['fondo_mejoramiento']
        else:
            row['fondo_mejoramiento'] = 0
        
        # Recalculate total height after adjusting fondo_mejoramiento
        row['total_height'] = sum(row[layer] for layer in layer_order if pd.notnull(row[layer]))
        
        # If still exceeding available depth, adjust layers from bottom up
        if row['total_height'] > available_depth:
            excess = row['total_height'] - available_depth
            for layer in reversed(layer_order):
                if pd.notnull(row[layer]) and excess > 0:
                    reduction = min(row[layer], excess)
                    row[layer] -= reduction
                    excess -= reduction
                    if excess <= 0:
                        break
        
        return row

    def adjust_layers(self, row, available_depth):
        """
        Ajusta las capas de relleno para que quepan en la profundidad disponible,
        ignorando ciertas capas superiores según el tipo de superficie (p.ej. 'superficie_lastre' en 'lastre').
        """
        # 1) Capas a ignorar por tipo de superficie (solo afectan al ajuste, no al valor real del row)
        IGNORED_LAYERS_BY_SURFACE = {
            'lastre': {'superficie_lastre'},
            # <- aquí decimos: en superficies tipo 'lastre', no cuentes esta capa en el ajuste
        }

        surface = row.get('sup_road', None)
        ignored_layers = IGNORED_LAYERS_BY_SURFACE.get(surface, set())

        # 2) Recolectar todas las capas posibles desde tu configuración
        all_layers = set(['sup_road'])
        for surface_type, layers in vg.tabla_relleno_suelo.items():
            if isinstance(layers, dict):
                all_layers.update(layers.keys())

        # 3) Orden de capas y limpieza de 'sup_road'
        layer_order = sorted(
            all_layers,
            key=lambda x: vg.tipos_relleno.index(x) if x in vg.tipos_relleno else float('inf')
        )
        if 'sup_road' in layer_order:
            layer_order.remove('sup_road')

        # 4) Trabajar con una copia local para el ajuste (no pisamos valores reales del row)
        row_adj = row.copy()

        # Poner en cero solo en la copia las capas ignoradas (p.ej. 'superficie_lastre' si sup_road == 'lastre')
        for lyr in ignored_layers:
            if lyr in row_adj and pd.notnull(row_adj[lyr]):
                row_adj[lyr] = 0.0

        # 5) Suma de capas superiores (excluye 'fondo_mejoramiento')
        upper_layers_sum = 0.0
        for lyr in layer_order:
            if lyr != 'fondo_mejoramiento':
                val = row_adj.get(lyr, 0.0)
                if pd.notnull(val):
                    upper_layers_sum += val

        # 6) Profundidad remanente para 'fondo_mejoramiento'
        remaining_depth = available_depth - upper_layers_sum

        if remaining_depth > 0 and row.get('sup_road') not in ['pasto']:
            row_adj['fondo_mejoramiento'] = remaining_depth * vg.tabla_relleno_suelo['relleno_mejoramiento']
        elif remaining_depth > 0 and row.get('sup_road') in ['pasto']:
            row_adj['fondo_mejoramiento'] = vg.tabla_relleno_suelo[row['sup_road']]['fondo_mejoramiento']
        else:
            row_adj['fondo_mejoramiento'] = 0.0

        # 7) Altura total (en la copia, que ya ignora las capas excluidas)
        total_height = 0.0
        for lyr in layer_order:
            val = row_adj.get(lyr, 0.0)
            if pd.notnull(val):
                total_height += val
        row['total_height'] = total_height  # guardamos el total calculado ignorando las capas excluidas

        # 8) Si excede, reducir de abajo hacia arriba (en la copia)
        if row['total_height'] > available_depth:
            excess = row['total_height'] - available_depth
            for lyr in reversed(layer_order):
                # OJO: no tocamos capas ignoradas (ya valen 0 en la copia)
                if lyr in ignored_layers:
                    continue
                val = row_adj.get(lyr, 0.0)
                if pd.notnull(val) and excess > 0:
                    reduction = min(val, excess)
                    row_adj[lyr] = val - reduction
                    excess -= reduction
                    if excess <= 0:
                        break

            # Recalcular total luego de reducir
            total_height = 0.0
            for lyr in layer_order:
                val = row_adj.get(lyr, 0.0)
                if pd.notnull(val):
                    total_height += val
            row['total_height'] = total_height

        # 9) Escribir de vuelta SOLO las capas ajustadas (no tocamos las ignoradas: mantienen sus valores originales en 'row')
        for lyr in layer_order:
            if lyr not in ignored_layers:
                row[lyr] = row_adj.get(lyr, row.get(lyr, 0.0))

        return row

    def calcular_area_trapecio(self, variable, base_inferior, base_superior, altura_total, altura_parcial, diametro, seccion, desde_arriba=True):
        """
        Calcula el área de un trapecio dada una altura parcial, medida desde arriba o desde abajo.
        
        Parámetros:
        base_inferior (float): Ancho de la base inferior del trapecio
        base_superior (float): Ancho de la base superior del trapecio
        altura_total (float): Altura total del trapecio
        altura_parcial (float): Altura parcial del trapecio
        desde_arriba (bool): Si True, la altura_parcial se mide desde arriba. Si False, desde abajo.
        
        Retorna:
        float: Área del trapecio
        """
        if desde_arriba:
            # Calcular el ancho a la altura parcial desde arriba
            ancho_parcial = base_superior - (base_superior - base_inferior) * (altura_parcial / altura_total)
            # Calcular el área del trapecio desde arriba
            area = (base_superior + ancho_parcial) * altura_parcial / 2
        else:
            if variable not in vg.vol_fondo:
                altura_parcial = altura_parcial + diametro
            # Calcular el ancho a la altura parcial desde abajo
            ancho_parcial = base_inferior + (base_superior - base_inferior) * (altura_parcial / altura_total)
            # Calcular el área del trapecio desde abajo
            area = (base_inferior + ancho_parcial) * (altura_parcial) / 2
            if variable not in vg.vol_fondo:
                filtro_circular = seccion == 'circular'
                filtro_rectangular = seccion == 'rectangular'
                
                h_circular = self.seccion_str2float(self.m_ramales_df['D_ext'][filtro_circular])
                area_circular = np.pi * (h_circular / 2.0) ** 2
                
                h_rectangular = self.seccion_str2float(self.m_ramales_df['D_ext'][filtro_rectangular])
                b_rectangular = self.seccion_str2float(self.m_ramales_df['D_ext'][filtro_rectangular], return_b=True)
                area_rectangular = h_rectangular * b_rectangular
                
                area_colector = np.empty(len(seccion))
                
                circular_index = filtro_circular.nonzero()[0]
                rectangular_index = filtro_rectangular.nonzero()[0]
                area_colector[circular_index] = area_circular
                area_colector[rectangular_index] = area_rectangular
                
                area = area - area_colector
                area = np.where(area < 0, 0, area)
        return area
    
    def get_fill_vol(self):
        """
        Calculate and adjust fill volumes for each road section.
        
        :return: DataFrame with adjusted fill volumes for each layer
        """
        
        if 'sup_road' in self.m_ramales_df.columns:
            # Initialize fill DataFrame with necessary columns
            self.fill_df = pd.DataFrame(columns=vg.tipos_relleno)
            
            # Preprocess sup_road column: lowercase, strip whitespace, replace spaces with underscores
            self.fill_df['sup_road'] = self.m_ramales_df['sup_road'].str.lower().str.strip().str.replace(' ', '_', regex=False)
            
            # Apply initial road surface mapping
            self.fill_df = self.fill_df.apply(lambda row: self.map_road_surface(row, vg.tabla_relleno_suelo), axis=1)
            
            # Calculate actual depths and available depths
            actual_depths = (self.m_ramales_df['HI'] + self.m_ramales_df['HF']) / 2
            pipe_heights = self.seccion_str2float(self.m_ramales_df['D_ext'])
            available_depths = actual_depths - pipe_heights
            
            # Adjust layers for each row based on available depth
            self.fill_df = self.fill_df.apply(lambda row: self.adjust_layers(row, available_depths.loc[row.name]), axis=1)
            
            # # Check for any rows still exceeding available depth after adjustment
            # excess_mask = self.fill_df['total_height'] > available_depths
            # excess_rows = self.fill_df[excess_mask]
            # if not excess_rows.empty:
            #     print(f"Warning: {len(excess_rows)} rows still exceed the available depth after adjustment.")
            #     print(excess_rows)
            
            # Add reference columns to the DataFrame
            self.fill_df['actual_depth'] = actual_depths
            self.fill_df['pipe_height'] = pipe_heights
            self.fill_df['available_depth'] = available_depths
            self.fill_df['seccion'] = self.m_ramales_df['Seccion']
            
            # ----------------------------------------------------------------------------------------
            altura_total = self.area_properties['profundidad'].to_numpy()
            base_inferior = self.area_properties['base_inferior'].to_numpy()
            base_superior = self.area_properties['base_superior'].to_numpy()
            length = self.area_properties['longitud'].to_numpy()
            seccion = self.fill_df['seccion'].to_numpy()
            columns_vol = []
            for rubro in vg.tipos_relleno:
                altura_parcial = self.fill_df[rubro].fillna(0).to_numpy()
                if vg.unidades_relleno[rubro] in ['m3']:
                    arr = self.calcular_area_trapecio(rubro, base_inferior, base_superior, altura_total, altura_parcial, pipe_heights, seccion, desde_arriba=True if 'superficie' in rubro else False) * length
                    self.fill_df[rubro + '_vol'] = arr
                    columns_vol.append(rubro + '_vol')
                elif vg.unidades_relleno[rubro] in ['m2']:
                    filtro = np.where(self.m_ramales_df['sup_road'].str.lower() == vg.grupo_relleno[rubro].lower(), 1, 0)
                    arr = length * (base_superior + 0.6) * filtro
                    if rubro in ['superficie_lastre']:
                        arr = arr * vg.tabla_relleno_suelo['lastre']['superficie_lastre']
                        self.fill_df[rubro + '_vol'] = arr
                        columns_vol.append(rubro + '_vol')
                    else:
                        self.fill_df[rubro + '_area'] = arr
                        columns_vol.append(rubro + '_area')
            
            df_fill = self.fill_df[columns_vol].fillna(0).sum(axis=0)
            
            # Ensure the columns to be summed exist, filled with 0 if they are missing
            df_fill['superficie_mejoramiento_vol'] = df_fill.get('superficie_mejoramiento_vol', pd.Series([0] * len(df_fill)))
            df_fill['fondo_mejoramiento_vol'] = df_fill.get('fondo_mejoramiento_vol', pd.Series([0] * len(df_fill)))
            
            df_fill['mejoramiento_vol'] = df_fill['superficie_mejoramiento_vol'] + df_fill['fondo_mejoramiento_vol']
            df_fill = df_fill.drop(['superficie_mejoramiento_vol', 'fondo_mejoramiento_vol'])
            
            df_fill = df_fill.to_frame()
            df_fill.columns = ['CANTIDAD']
            df_fill['SECTION'] = np.nan
            
            map_dict = {'superficie_lastre_vol': 'REPOSICION DE LASTRE', 'superficie_adoquin_area': 'REMOCION Y COLOCACION DE ADOQUIN', 'superficie_asfalto_vol': 'CORTE , ROTURA Y REPOSICION DE PAVIMENTO FLEXIBLE', 'superficie_hormigon_pavimento_vol': 'CORTE , ROTURA Y REPOSICION DE PAVIMENTO RIGIDO', 'superficie_hormigon_vereda_vol': 'CORTE , ROTURA Y REPOSICION DE VEREDAS'}
            reposicion_lista = list(map_dict.keys())

            df_fill['SECTION'] = df_fill.index.to_series().map(map_dict).fillna('RELLENO Y DESALOJO')
            
            df_rotura = df_fill.loc[reposicion_lista].copy()
            df_rotura.index = ['rotura_' + _ for _ in df_rotura.index.to_list()]
            df_rotura = df_rotura.drop(['rotura_superficie_lastre_vol']) # solo se remueve este porque este no tiene un suministro y remocion ( no se remueve primero el lastre para arreglar la via como en el hormigon o veredes etc.)

            df_acero_reposicion = pd.DataFrame([df_fill['CANTIDAD']['superficie_hormigon_vereda_vol'] * 90, df_fill['CANTIDAD']['superficie_hormigon_pavimento_vol']], index=['malla_electrosoldada', 'peso_acero'], columns=['CANTIDAD'])
            df_acero_reposicion['SECTION'] = ['CORTE , ROTURA Y REPOSICION DE VEREDAS', 'CORTE , ROTURA Y REPOSICION DE PAVIMENTO RIGIDO']

            rendimiento_motoniveladora = 40 # 40 m3/hora
            horas_equipos =  int(df_fill['CANTIDAD']['superficie_lastre_vol'] / rendimiento_motoniveladora)
            df_maquinaria_arreglo_lastre= pd.DataFrame([horas_equipos, horas_equipos, horas_equipos ], index=['horas_motoniveladora', 'horas_rodillo', 'horas_tanquero_agua'], columns=['CANTIDAD'])
            df_maquinaria_arreglo_lastre['SECTION'] = ['REPOSICION DE LASTRE', 'REPOSICION DE LASTRE', 'REPOSICION DE LASTRE']

            return df_fill, pd.concat([df_rotura, df_acero_reposicion, df_maquinaria_arreglo_lastre])

    
    # -------------------------------------------------------------------------------------------------------------------
    def soil_improvement(self, N60_array, D_array, base_width_array, overlap=0.3):
        depth_array = np.zeros_like(N60_array, dtype=float)
        geotextil_length_array = np.zeros_like(N60_array, dtype=float)
        for i, (N60, D, base_width) in enumerate(zip(N60_array, D_array, base_width_array)):
            if N60 < 6:
                depth = 1.5 * D
                depth = 1 if depth > 1 else depth
                depth = 0.2 if depth < 0.2 else depth
                
                if depth <= 0.5:
                    geotextil_length = (2 * depth + 2 * base_width + overlap)
                else:
                    
                    full_layers = int(depth // 0.5)
                    geotextil_length = full_layers * (2 * 0.5 + 2 * base_width + overlap)
                    
                    remaining_depth = (depth - 0.5 * full_layers) % 0.5
                    if remaining_depth > 0:
                        geotextil_length += (2 * remaining_depth + 2 * base_width + overlap)
            
            elif 6 <= N60 < 8:
                depth = 1 * D
                depth = 1 if depth > 1 else depth
                depth = 0.2 if depth < 0.2 else depth
                geotextil_length = 0
            
            elif 8 <= N60 < 10:
                depth = 0.5 * D
                depth = 1 if depth > 1 else depth
                depth = 0.2 if depth < 0.2 else depth
                geotextil_length = 0
            
            else:  # N60 >= 10
                depth = 0
                geotextil_length = 0
            
            depth_array[i] = depth
            geotextil_length_array[i] = geotextil_length
        return depth_array - 0.1, geotextil_length_array
    
    def cement_grout_injection(self, N60_array, top_width_array, longitud_array, profunidad_array):
        
        filtro_N60 = N60_array < 3
        
        ancho_aplicacion = top_width_array
        largo_aplicacion = longitud_array
        
        distancia_entre_inyecciones = 1
        diametro_perforacion = 0.1
        cantidad_perforaciones_ancho = np.round(ancho_aplicacion / distancia_entre_inyecciones, 0)
        cantidad_perforaciones_largo = np.round(largo_aplicacion / distancia_entre_inyecciones, 0)
        cantidad_perforaicones = cantidad_perforaciones_ancho * cantidad_perforaciones_largo
        
        volumen_perforacion_unitario = (profunidad_array + profunidad_array * 0.3) * np.pi * (diametro_perforacion / 2) ** 2
        peso_lechada = 2  # Ton/m3
        peso_perforacion_unitario = volumen_perforacion_unitario * peso_lechada
        peso_perforaciones = cantidad_perforaicones * peso_perforacion_unitario
        porcentaje_vacios = 0.25
        peso_perforaciones = peso_perforaciones * filtro_N60 * (1 + porcentaje_vacios)
        
        return peso_perforaciones
    
    
    def vol_replacement_excavation(self):
        trench_depths = self.area_properties['profundidad']
        replacement_depths = self.soil_replacement_height
        
        total_depths = trench_depths + replacement_depths
        results = []
        for start, end in np.unique(self.depth_ranges, axis=0):
            vol = np.clip(np.minimum(end, total_depths) - np.maximum(start, trench_depths), 0, end - start)
            results.append(vol)
        column_labels = [f"{start}-{end if not np.isinf(end) else 'inf'}" for start, end in np.unique(self.depth_ranges, axis=0)]
        return pd.DataFrame(np.array(results).T, columns=column_labels)
    
    def get_soil_improvement(self):
        # Check if 'N60' column exists in the DataFrame
        if 'N60' in self.m_ramales_df.columns:
            # Extract N60 values and convert to numpy array
            N60_array = self.m_ramales_df['N60'].to_numpy()
            # Convert external diameter values to float and store in array
            diameter_array = self.seccion_str2float(self.m_ramales_df['D_ext'])
            # Get the base width from area properties
            base_width = self.area_properties['base_inferior']
            top_width = self.area_properties['base_superior']
            longitud = self.area_properties['longitud']
            profundidad = self.area_properties['profundidad']
            
            #get grout injections
            peso_inyecciones_lechada = self.cement_grout_injection(N60_array, top_width, longitud, profundidad )
            
            # Calculate soil replacement height and geotextile length
            soil_replacement_height, geotextil_length = self.soil_improvement(N60_array, diameter_array, base_width)
            self.soil_replacement_height = soil_replacement_height
            # Calculate soil replacement volume and geotextile area
            soil_replacement_vol = soil_replacement_height * self.m_ramales_df['L'].to_numpy()
            geotextil_area = geotextil_length * self.m_ramales_df['L'].to_numpy()
            
            # Create a new DataFrame for soil replacement data
            soil_replacement = pd.DataFrame(index=self.m_ramales_df.index)
            soil_replacement['relleno_grava'] = soil_replacement_vol * 0.5
            soil_replacement['relleno_piedra'] = soil_replacement_vol * 0.5
            soil_replacement['area_geotextil'] = geotextil_area
            soil_replacement['inyenccion_lechada'] = peso_inyecciones_lechada
            
            # Store soil replacement data
            self.soil_replacement = soil_replacement.copy()
            soil_replacement = soil_replacement.sum(axis=0)
            
            # Check construction methods for manual and machine trenching
            condicion_zanja_mano = self.m_ramales_df['metodo_constructivo'].str.contains("|".join(self.metodos_constructivos_mano), case=False, na=False).to_numpy()
            condicion_zanja_maquina = self.m_ramales_df['metodo_constructivo'].str.contains("|".join(self.metodos_constructivos_maquina), case=False, na=False).to_numpy()
            
            # Calculate replacement excavation volume
            height_replacement = self.vol_replacement_excavation()
            
            # Calculate area and volume
            area = height_replacement.multiply(self.area_properties['base_inferior'], axis=0)
            vol = area.multiply(self.area_properties['longitud'], axis=0)
            
            # Process volumes for machine and manual trenching
            cantidades_maquina = self.process_vol_rows(vol[condicion_zanja_maquina], self.tabla_tipo_suelo_maquina)
            cantidades_mano = self.process_vol_rows(vol[condicion_zanja_mano], self.tabla_tipo_suelo_mano)
            
            # Combine all calculated quantities
            df_soil_replacement = pd.concat([cantidades_mano, cantidades_maquina, soil_replacement])
            # Format the final DataFrame
            df_soil_replacement = df_soil_replacement.to_frame()
            df_soil_replacement.columns = ['CANTIDAD']
            df_soil_replacement['SECTION'] = 'MEJORAMIENTO DE FONDO DE ZANJA'
            
            return df_soil_replacement
        
        else:
            return pd.Series([])
    
    # -------------------------------------------------------------------------------------------------------------------
    def get_material_removal(self, vol_df):
        
        mano_columns = [_ for _ in vol_df.index if 'manual' in _]
        maquina_columns = [_ for _ in vol_df.index if 'mecanica' in _]
        vol_mano = vol_df.loc[mano_columns]['CANTIDAD'].sum()
        vol_maquina = vol_df.loc[maquina_columns]['CANTIDAD'].sum()
        vol_desalojo = (vol_mano + vol_maquina) * (1 + self.porcentage_esponjamiento) * self.distancia_desalojo
        
        df_material_removal = pd.DataFrame(index=self.m_ramales_df.index)
        
        # verificar si en los metodos constructivos hay zanja mano
        condicion_zanja_mano = self.m_ramales_df['metodo_constructivo'].str.contains("|".join(self.metodos_constructivos_mano), case=False, na=False).to_numpy()
        
        # volumen de trasnporte desde rellenos
        column_names = [_ for _ in self.fill_df.columns if '_vol' in _]
        removal_fill = self.fill_df[column_names]
        df_material_removal['vol_mano_fill'] = removal_fill[condicion_zanja_mano].sum(axis=1)
        
        if 'N60' in self.m_ramales_df.columns:
            # volumen de mejoramiento de fondo de zanja
            replacement_soil = self.soil_replacement.reset_index(drop=True)
            df_material_removal['vol_mano_replacement'] = replacement_soil[condicion_zanja_mano].sum(axis=1)
        else:
            df_material_removal['vol_mano_replacement'] = 0
        
        columns_mano = [_ for _ in df_material_removal.columns if 'mano' in _]
        vol_mano_puesto_en_obra = df_material_removal[columns_mano].sum().sum()
        
        # volumen de tuberia
        filtro_circular = self.m_ramales_df['Seccion'] == 'circular'
        filtro_rectangular = self.m_ramales_df['Seccion'] == 'rectangular'
        
        h_circular = self.seccion_str2float(self.m_ramales_df['D_ext'][filtro_circular])
        area_circular = np.pi * (h_circular / 2.0) ** 2
        vol_circular = area_circular * self.m_ramales_df['L'][filtro_circular]
        
        h_rectangular = self.seccion_str2float(self.m_ramales_df['D_ext'][filtro_rectangular])
        b_rectangular = self.seccion_str2float(self.m_ramales_df['D_ext'][filtro_rectangular], return_b=True)
        area_rectangular = h_rectangular * b_rectangular
        vol_rectangular = area_rectangular * self.m_ramales_df['L'][filtro_rectangular]
        
        vol_colectores = vol_rectangular.sum() + vol_circular.sum()
        
        vol_sustitucion_suelo_zanja = vol_colectores + removal_fill.sum().sum()
        vol_cut = self.vol_rows.sum().sum()
        vol_compactado = max(1, np.abs(vol_cut - vol_sustitucion_suelo_zanja))
        
        acarreo_desalojo = pd.Series([(vol_mano + vol_mano_puesto_en_obra) * self.distancia_acarreo_manual, vol_maquina, vol_desalojo, vol_compactado], index=['acarreo_manual', 'acarreo_mecanico', 'desalojo', 'relleno_compactado'])
        
        acarreo_desalojo = acarreo_desalojo.to_frame()
        acarreo_desalojo.columns = ['CANTIDAD']
        acarreo_desalojo['SECTION'] = 'RELLENO Y DESALOJO'
        return acarreo_desalojo
    
    # -------------------------------------------------------------------------------------------------------------------
    def get_other_quantities(self):
        
        if 'sup_road' in self.m_ramales_df.columns:
            # desbroce
            filtro = self.m_ramales_df['sup_road'].str.lower() == 'pasto'
            filtro = filtro.to_numpy()
            if len(filtro.nonzero()[0]) > 0:
                desbroce = self.m_ramales_df['L'][filtro] * self.area_properties['base_superior'][filtro] + 1
            else:
                desbroce = self.m_ramales_df['L'] * (self.area_properties['base_superior'] + 1) * self.porcentage_desbroce
        
        else:
            desbroce = self.m_ramales_df['L'] * (self.area_properties['base_superior'] + 1) * self.porcentage_desbroce
        
        abatimiento_nivel_freatico = (self.longitud_total / 35) * 4
        replanteo_nivelacion = self.longitud_total / 1000
        
        other_quantities = pd.Series([np.sum(desbroce), replanteo_nivelacion, abatimiento_nivel_freatico, abatimiento_nivel_freatico], index=['desbroce', 'replanteo', 'abatimiento_nivel_freatico', 'abatimiento_agua_residual'])
        other_quantities = other_quantities.to_frame()
        other_quantities.columns = ['CANTIDAD']
        other_quantities['SECTION'] = ['DESBROCE Y LIMPIEZA', 'REPLANTEO Y NIVELACION', 'BOMBEO DE AGUA', 'BOMBEO DE AGUA']
        
        return other_quantities


class CantidadesPerforacionHorizontalDirigida:
    def __init__(self, parameters_dict):
        
        self.parameters_dict = parameters_dict
        self.distancia_desalojo = self.parameters_dict['distancia_desalojo']
        
        # get vector path
        self.vector_path = parameters_dict['vector_path']
        try:
            self.m_ramales_df = gpd.read_file(self.vector_path, engine='pyogrio')
        except:
            self.m_ramales_df = gpd.read_file(self.vector_path)
        
        # filter out only requiered column
        filtro_dict = parameters_dict.get('filtro')
        if filtro_dict:
            filtro_column = filtro_dict.get('column')
            filtro_value = filtro_dict.get('value')
            
            if filtro_column and filtro_value:
                # Convert filtro_value to a list if it's a string of comma-separated values
                if isinstance(filtro_value, str):
                    filtro_value = [x.strip() for x in filtro_value.split(',')]
                
                # Apply the filter using isin to handle lists of values
                filtro = self.m_ramales_df[filtro_column].isin(filtro_value)
                self.m_ramales_df = self.m_ramales_df.loc[filtro]
        
        # volumen de tramos nuevos
        filtro_nuevo = self.m_ramales_df['Estado'] == 'nuevo'
        self.m_ramales_df = self.m_ramales_df.loc[filtro_nuevo]
        
        self.longitud_total = self.m_ramales_df['L'].sum()
        
        # tramos solo  zanja
        self.metodos_constructivos = ['perforacion horizontal dirigida']
        filtro_metodo_constructivo = self.m_ramales_df['metodo_constructivo'].str.contains("|".join(self.metodos_constructivos), case=False, na=False)
        self.m_ramales_df = self.m_ramales_df.loc[filtro_metodo_constructivo]
        self.m_ramales_df = self.m_ramales_df.reset_index(drop=True)
    
    def seccion_str2float(self, arr, return_b=False):
        """
        :param arr: array of string mix of circular and rectangular sections
        :return: array of diameter of circular sections and height dimensions of rectangular sections
        """
        # Convert input to pandas Series if it's not already
        arr_series = pd.Series(arr)
        
        # Check if there are any 'x' in the array
        if arr_series.str.contains('x').any():
            # Split base and height when possible
            split_arr = arr_series.str.split('x', expand=True)
            b = split_arr[0].fillna(0).to_numpy().astype(str)
            h = split_arr[1].fillna(0).to_numpy().astype(str)
            
            # Get circular and rectangular indexes
            circular_index = np.char.equal(h, '0').nonzero()[0]
            rectangular_index = np.char.not_equal(h, '0').nonzero()[0]
            
            # Zeros array
            out = np.zeros(shape=len(b), dtype=float)
            
            # Fill zeros array
            out[circular_index] = b[circular_index].astype(float)
            
            if return_b:
                out[rectangular_index] = b[rectangular_index].astype(float)
            else:
                out[rectangular_index] = h[rectangular_index].astype(float)
        else:
            # If there are no 'x', treat all as circular sections
            out = arr_series.astype(float).to_numpy()
        
        return out
    
    def get_length_metodo_constructivo(self):
        
        # Filter for new or to be repaired pipes
        filtro_nuevo = np.logical_and(self.m_ramales_df['Estado'] == 'nuevo', self.m_ramales_df['metodo_constructivo'] != 'existente')
        metodo_constructivo_df = self.m_ramales_df.loc[filtro_nuevo]
        
        metodos = metodo_constructivo_df['metodo_constructivo'].unique()
        secciones = metodo_constructivo_df['Seccion'].unique()
        materiales = metodo_constructivo_df['Material'].unique()
        # Prebuild the result dictionary
        res = {metodo: {seccion: {material: None for material in materiales} for seccion in secciones} for metodo in metodos}
        for metodo in metodos:
            filtro_metodo = metodo_constructivo_df['metodo_constructivo'] == metodo
            metodo_df = metodo_constructivo_df.loc[filtro_metodo]
            for seccion in secciones:
                filtro_seccion = metodo_df['Seccion'] == seccion
                seccion_df = metodo_df.loc[filtro_seccion]
                for material in materiales:
                    filtro_material = seccion_df['Material'] == material
                    material_df = seccion_df.loc[filtro_material]
                    if material_df.empty:
                        continue
                    # Group by 'D_ext' and sum up the 'L' column
                    s = material_df.groupby('D_ext')['L'].sum()
                    # Sort the Series index
                    sorted_index = natsorted(s.index, alg=ns.FLOAT)
                    # Reindex the Series based on the sorted index
                    s_sorted = s.reindex(sorted_index)
                    res[metodo][seccion][material] = s_sorted
        # Filter out entries with None values
        filtered_res = {metodo: {seccion: {material: lengths for material, lengths in materials.items() if lengths is not None} for seccion, materials in secciones.items() if any(lengths is not None for lengths in materials.values())} for metodo, secciones in res.items() if any(any(lengths is not None for lengths in materials.values()) for materials in secciones.values())}
        # Convert to a DataFrame readable by people to Excel
        rows = []
        for metodo, secciones in filtered_res.items():
            for seccion, materiales in secciones.items():
                for material, lengths in materiales.items():
                    for d_ext, length in lengths.items():
                        rows.append([metodo, seccion, material, d_ext, length])
        df = pd.DataFrame(rows, columns=['metodo_constructivo', 'Seccion', 'Material', 'D_ext', 'L'])
        
        self.cantidades_colectores = df
        
        return df
    
    def get_soil_vol(self, total_volume):
        results = []
        for soil_type, percentage in vg.tabla_tipo_suelo["tipo_suelo_mano"].items():
            index = f"{soil_type}"
            volume = total_volume * percentage
            results.append((index, volume))
        # Convert results to Series
        result_series = pd.Series(dict(results))
        return result_series
    
    def get_length_phd(self):
        
        df = self.m_ramales_df
        if not df.empty:
            self.cantidades_colectores = self.get_length_metodo_constructivo()
            
            # Filter for circular pipes
            filtro_circular = self.cantidades_colectores['Seccion'] == 'circular'
            
            # Create a unique index by combining external diameter and material
            indice_circular = self.cantidades_colectores[filtro_circular]['D_ext'] + '-' + self.cantidades_colectores[filtro_circular]['Material']
            
            # Create a new DataFrame with lengths of circular pipes, indexed by the unique combination
            cantidades_pipes = pd.DataFrame(self.cantidades_colectores[filtro_circular]['L'].to_numpy(), index=indice_circular, columns=['CANTIDAD'])
            
            # Group by the unique index and sum the lengths
            cantidades_pipes = cantidades_pipes['CANTIDAD'].groupby(cantidades_pipes.index).sum()
            
            # cantidad de lodo
            vol_lodo = np.sum(np.pi * np.sqrt(self.seccion_str2float(self.cantidades_colectores[filtro_circular]['D_ext']) / 2.0) * self.cantidades_colectores[filtro_circular]['L'].to_numpy())
            cantidades_pipes['desalojo_lodo'] = vol_lodo * self.distancia_desalojo
            cantidades_pipes['interferencias'] = self.cantidades_colectores['L'].sum()
            
            # Convert the series back to a DataFrame
            cantidades_pipes = cantidades_pipes.to_frame()
            
            # Add a new column 'SECTION' with a constant value
            cantidades_pipes['SECTION'] = "PERFORACION HORIZONTAL DIRIGIDA"
            
            # Return the final DataFrame
            return cantidades_pipes
        
        else:
            return pd.DataFrame([])
    
    def get_trinchera(self):
        df = self.m_ramales_df
        if not df.empty:
            
            distancia_trinchera = vg.parametros_cantidades['distancia_trinchera_perforacion_horizontal_dirigida']
            seccion_minima_trinchera = vg.parametros_cantidades['seccion_minima_trinchera']
            e_hormigon_pobre = vg.parametros_cantidades['espesor_hormigon pobre']
            
            pz_depths = []
            grupos = df.groupby('Ramal')
            for _, grupo in grupos:
                lengths = np.cumsum(grupo['L'])
                depths = np.max([grupo['HF'], grupo['HI']], axis=0)
                distance = lengths + depths
                multiples = np.arange(0, distance.max() + distancia_trinchera, distancia_trinchera)
                indices = np.where(np.abs(distance.to_numpy()[:, np.newaxis] - multiples).argmin(axis=0) - 1 < 0, 0, np.abs(distance.to_numpy()[:, np.newaxis] - multiples).argmin(axis=0) - 1)
                if isinstance(depths[indices], float):
                    arr = [depths[indices]]
                else:
                    arr = depths[indices]
                
                pz_depths.append(arr)
            pz_depths = np.concatenate(pz_depths)
            
            # get depth
            h = pz_depths
            
            ancho_interno = np.full(len(h), fill_value=seccion_minima_trinchera)
            largo_interno = np.full(len(h), fill_value=seccion_minima_trinchera)
            
            # ---------------------------------------------------------------------------
            replantillo = ancho_interno * largo_interno
            hormigon_pobre = replantillo * e_hormigon_pobre
            cut_vol = self.get_soil_vol(np.sum(replantillo * h))
            acarreo_manual = cut_vol * 0.2 * 20
            acarreo_mecanico = cut_vol * 0.8
            desalojo = cut_vol * self.distancia_desalojo
            relleno_compactado = replantillo * h * 0.3
            material_mejoramiento = replantillo * h * 0.7
            entibado = h * (ancho_interno) * 2 + (largo_interno) * 2
            
            cut_vol['relleno_compactado'] = relleno_compactado.sum()
            cut_vol['material_mejoramiento'] = material_mejoramiento.sum()
            cut_vol['replantillo'] = replantillo.sum()
            cut_vol['hormigon_pobre'] = hormigon_pobre.sum()
            cut_vol['entibado'] = entibado.sum()
            cut_vol['acarreo_manual'] = acarreo_manual.sum()
            cut_vol['acarreo_mecanico'] = acarreo_mecanico.sum()
            cut_vol['desalojo'] = desalojo.sum()
            
            cut_vol = cut_vol.to_frame()
            cut_vol.columns = ['CANTIDAD']
            cut_vol['SECTION'] = "TRINCHERAS PERFORACION HORIZONTAL DIRIGIDA"
            
            return cut_vol
        else:
            return pd.DataFrame([])
    
    def get_phd(self):
        df_phd = self.get_length_phd()
        df_trinchera = self.get_trinchera()
        
        return pd.concat([df_phd, df_trinchera])


class CantidadesTunel:
    def __init__(self, parameters_dict):
        
        self.parameters_dict = parameters_dict
        self.distancia_desalojo = self.parameters_dict['distancia_desalojo']
        
        # get vector path
        self.vector_path = parameters_dict['vector_path']
        try:
            self.m_ramales_df = gpd.read_file(self.vector_path, engine='pyogrio')
        except:
            self.m_ramales_df = gpd.read_file(self.vector_path)
        
        # filter out only requiered column
        filtro_dict = parameters_dict.get('filtro')
        if filtro_dict:
            filtro_column = filtro_dict.get('column')
            filtro_value = filtro_dict.get('value')
            
            if filtro_column and filtro_value:
                # Convert filtro_value to a list if it's a string of comma-separated values
                if isinstance(filtro_value, str):
                    filtro_value = [x.strip() for x in filtro_value.split(',')]
                
                # Apply the filter using isin to handle lists of values
                filtro = self.m_ramales_df[filtro_column].isin(filtro_value)
                self.m_ramales_df = self.m_ramales_df.loc[filtro]
        
        # volumen de tramos nuevos
        filtro_nuevo = self.m_ramales_df['Estado'] == 'nuevo'
        self.m_ramales_df = self.m_ramales_df.loc[filtro_nuevo]
        
        # tramos
        self.metodos_constructivos_tunel = ['tunel']
        filtro_metodo_constructivo = self.m_ramales_df['metodo_constructivo'].str.contains("|".join(self.metodos_constructivos_tunel), case=False, na=False)
        self.m_ramales_df = self.m_ramales_df.loc[filtro_metodo_constructivo]
        
        self.m_ramales_df = self.m_ramales_df.reset_index(drop=True)
    
    def seccion_str2float(self, arr, return_b=False):
        """
        :param arr: array of string mix of circular and rectangular sections
        :return: array of diameter of circular sections and height dimensions of rectangular sections
        """
        # Convert input to pandas Series if it's not already
        arr_series = pd.Series(arr)
        
        # Check if there are any 'x' in the array
        if arr_series.str.contains('x').any():
            # Split base and height when possible
            split_arr = arr_series.str.split('x', expand=True)
            b = split_arr[0].fillna(0).to_numpy().astype(str)
            h = split_arr[1].fillna(0).to_numpy().astype(str)
            
            # Get circular and rectangular indexes
            circular_index = np.char.equal(h, '0').nonzero()[0]
            rectangular_index = np.char.not_equal(h, '0').nonzero()[0]
            
            # Zeros array
            out = np.zeros(shape=len(b), dtype=float)
            
            # Fill zeros array
            out[circular_index] = b[circular_index].astype(float)
            
            if return_b:
                out[rectangular_index] = b[rectangular_index].astype(float)
            else:
                out[rectangular_index] = h[rectangular_index].astype(float)
        else:
            # If there are no 'x', treat all as circular sections
            out = arr_series.astype(float).to_numpy()
        
        return out
    
    def get_length_metodo_constructivo(self):
        
        # Filter for new or to be repaired pipes
        filtro_nuevo = np.logical_and(self.m_ramales_df['Estado'] == 'nuevo', self.m_ramales_df['metodo_constructivo'] != 'existente')
        metodo_constructivo_df = self.m_ramales_df.loc[filtro_nuevo]
        
        metodos = metodo_constructivo_df['metodo_constructivo'].unique()
        secciones = metodo_constructivo_df['Seccion'].unique()
        materiales = metodo_constructivo_df['Material'].unique()
        # Prebuild the result dictionary
        res = {metodo: {seccion: {material: None for material in materiales} for seccion in secciones} for metodo in metodos}
        for metodo in metodos:
            filtro_metodo = metodo_constructivo_df['metodo_constructivo'] == metodo
            metodo_df = metodo_constructivo_df.loc[filtro_metodo]
            for seccion in secciones:
                filtro_seccion = metodo_df['Seccion'] == seccion
                seccion_df = metodo_df.loc[filtro_seccion]
                for material in materiales:
                    filtro_material = seccion_df['Material'] == material
                    material_df = seccion_df.loc[filtro_material]
                    if material_df.empty:
                        continue
                    # Group by 'D_ext' and sum up the 'L' column
                    s = material_df.groupby('D_ext')['L'].sum()
                    # Sort the Series index
                    sorted_index = natsorted(s.index, alg=ns.FLOAT)
                    # Reindex the Series based on the sorted index
                    s_sorted = s.reindex(sorted_index)
                    res[metodo][seccion][material] = s_sorted
        # Filter out entries with None values
        filtered_res = {metodo: {seccion: {material: lengths for material, lengths in materials.items() if lengths is not None} for seccion, materials in secciones.items() if any(lengths is not None for lengths in materials.values())} for metodo, secciones in res.items() if any(any(lengths is not None for lengths in materials.values()) for materials in secciones.values())}
        # Convert to a DataFrame readable by people to Excel
        rows = []
        for metodo, secciones in filtered_res.items():
            for seccion, materiales in secciones.items():
                for material, lengths in materiales.items():
                    for d_ext, length in lengths.items():
                        rows.append([metodo, seccion, material, d_ext, length])
        df = pd.DataFrame(rows, columns=['metodo_constructivo', 'Seccion', 'Material', 'D_ext', 'L'])
        
        self.cantidades_colectores = df
        
        return df
    
    def get_soil_vol(self, total_volume):
        results = []
        for soil_type, percentage in vg.tabla_tipo_suelo["tipo_suelo_mano"].items():
            index = f"{soil_type}"
            volume = total_volume * percentage
            results.append((index, volume))
        # Convert results to Series
        result_series = pd.Series(dict(results))
        return result_series
    
    def get_vol_tunel(self):
        
        df = self.m_ramales_df
        if not df.empty:
            self.cantidades_colectores = self.get_length_metodo_constructivo()
            
            # espesores tunel
            e_pared_tunel = vg.parametros_cantidades['espesor_pared_tunel']
            e_fondo_tunel = vg.parametros_cantidades['espesor_fondo_tunel']
            cuantia_acero_tunel = vg.parametros_cantidades['cuantia_acero_tunel']
            distancia_sostenimiento = vg.parametros_cantidades['distancia_sostenimiento_tunel']
            
            # dimensiones tunel
            h = self.seccion_str2float(self.cantidades_colectores['D_ext'])
            b = self.seccion_str2float(self.cantidades_colectores['D_ext'], return_b=True)
            
            # vol estructura
            area_boveda = ((np.pi * ((b / 2) + e_pared_tunel) ** 2) - (np.pi * (b / 2) ** 2)) / 2.0
            area_pared = (h - b / 2) * e_pared_tunel * 2
            area_fondo = (b + 2 * e_pared_tunel) * e_fondo_tunel
            
            vol_estructura = (area_boveda + area_pared + area_fondo) * self.cantidades_colectores['L']
            peso_acero = vol_estructura * cuantia_acero_tunel
            
            # 6.1 m sostenimiento (perimetro) -- > 30 kg
            pero_acero_sostenimiento = (2 * np.pi * (b / 2 + e_pared_tunel) + 2 * (h - b / 2 + e_fondo_tunel)) * (30 / 6.1) * 1.1
            pero_acero_sostenimiento = pero_acero_sostenimiento * np.round((self.cantidades_colectores['L'] * distancia_sostenimiento), 0) + 1
            
            # vol excavacion bodeva - colector
            area_superior = (np.pi * ((b / 2) + e_pared_tunel + 0.15) ** 2) / 2.0
            area_inferior = (h - b / 2 + e_fondo_tunel + 0.15) * (b + 2 * e_pared_tunel + 0.15)
            area_total = area_superior + area_inferior
            vol_excavacion = area_total * self.cantidades_colectores['L']
            volumes = self.get_soil_vol(vol_excavacion.sum())
            acarreo_manual = vol_excavacion * 0.2 * 20
            acarreo_mecanico = vol_excavacion * 0.8
            desalojo = vol_excavacion * self.distancia_desalojo
            
            # superficie de bodeva - colector
            superficie_boveda_externa = 2 * np.pi * (b / 2 + e_pared_tunel)
            superficie_colector_externa = (h + e_fondo_tunel) * 2
            superficie_externa = (superficie_boveda_externa + superficie_colector_externa) * self.cantidades_colectores['L']
            
            # superficie interna de bodeva - colector
            superficie_boveda_interna = 2 * np.pi * (b / 2) * self.cantidades_colectores['L']
            superficie_colector_interna = ((h * 2) + b) * self.cantidades_colectores['L']
            
            # area inferior base
            area_inferior = (b + 2 * e_pared_tunel + 0.15 * 2) * self.cantidades_colectores['L']
            
            # cinta PVC
            cinta = 4 * self.cantidades_colectores['L']
            
            # longitud total
            replanteo = self.cantidades_colectores['L']
            
            # df out
            df = volumes.copy()
            df['replanteo'] = replanteo.sum()
            df['entibado'] = superficie_externa.sum()
            df['encofrado_arco'] = superficie_boveda_interna.sum()
            df['encofrado_recto'] = superficie_colector_interna.sum()
            df['hormigon_pobre'] = area_inferior.sum() * 0.07
            df['hormigon_estructura'] = vol_estructura.sum()
            df['peso_acero_estructura'] = peso_acero.sum()
            df['peso_acero_sostenimiento'] = pero_acero_sostenimiento.sum()
            df['juntas_impermeables'] = cinta.sum()
            df['drenes'] = np.sum(self.cantidades_colectores['L'].sum() * np.round((b / 2) + 1))  # dren cada dos metros
            df['acarreo_manual'] = acarreo_manual.sum()
            df['acarreo_mecanico'] = acarreo_mecanico.sum()
            df['desalojo'] = desalojo.sum()
            
            df = df.to_frame()
            df.columns = ['CANTIDAD']
            df['SECTION'] = 'TUNEL'
            
            return df
        else:
            return pd.DataFrame([])
    
    def get_pozo_avance(self):
        df = self.m_ramales_df
        if not df.empty:
            
            distancia_pozo_avance = vg.parametros_cantidades['distancia_pozo_avance']
            seccion_minima_pozo = vg.parametros_cantidades['seccion_minima_pozo_avance']
            
            pz_depths = []
            grupos = df.groupby('Ramal')
            for _, grupo in grupos:
                lengths = np.cumsum(grupo['L'])
                depths = np.max([grupo['HF'], grupo['HI']], axis=0)
                distance = lengths + depths
                # Step 2: Generate multiples of 60 up to the maximum length
                multiples = np.arange(0, distance.max() + distancia_pozo_avance, distancia_pozo_avance)

                indices = np.where(np.abs(distance.to_numpy()[:, np.newaxis] - multiples).argmin(axis=0) - 1 < 0, 0, np.abs(distance.to_numpy()[:, np.newaxis] - multiples).argmin(axis=0) - 1)
                if isinstance(depths[indices], float):
                    arr = [depths[indices]]
                else:
                    arr = depths[indices]
                
                pz_depths.append(arr)
            pz_depths = np.concatenate(pz_depths)
            
            # get depth
            h = pz_depths
            
            # espesores de pared
            e_pared = vg.parametros_cantidades['espesor_pared_pozo']
            e_fondo = vg.parametros_cantidades['espesor_fondo_pozo']
            e_tapa = vg.parametros_cantidades['espesor_tapa_pozo']
            e_hormigon_pobre = vg.parametros_cantidades['espesor_hormigon pobre']
            cuantia_acero = vg.parametros_cantidades['cuantia_acero_pozo']
            
            ancho_interno = np.full(len(h), fill_value=seccion_minima_pozo)
            largo_interno = np.full(len(h), fill_value=seccion_minima_pozo)
            
            ancho_externo = ancho_interno + (2 * e_pared)
            largo_externo = largo_interno + (2 * e_pared)
            area_externa = ancho_externo * largo_externo
            
            area_interna = ancho_interno * largo_interno
            
            area_paredes = area_externa - area_interna
            area_seccion = area_paredes
            vol_seccion = area_seccion * h
            
            # ---------------------------------------------------------------------------
            vol_tapa = e_tapa * ancho_externo * largo_externo
            vol_fondo = e_fondo * ancho_externo * largo_externo
            vol_pz = vol_seccion + vol_tapa + vol_fondo
            peso_acero = vol_pz * cuantia_acero
            replantillo = (ancho_externo + 1) * (largo_externo + 1)
            hormigon_pobre = replantillo * e_hormigon_pobre
            encofrado = (area_externa * h) + (area_interna * h) + (area_externa + area_interna * 2) + (e_tapa * (largo_interno * 2 + ancho_externo * 2))
            cut_vol = self.get_soil_vol(np.sum(replantillo * h))
            tapas = len(h)
            estribos_varilla_grada = round(np.sum(h / 0.3).sum(), 0)
            acarreo_manual = cut_vol * 0.2 * 20
            acarreo_mecanico = cut_vol * 0.8
            desalojo = cut_vol * self.distancia_desalojo
            relleno_compactado = (replantillo - area_externa) * h * 0.3
            material_mejoramiento = (replantillo - area_externa) * h * 0.7
            entibado = h * (ancho_externo + 1) * 2 + (largo_externo + 1) * 2
            
            cut_vol['relleno_compactado'] = relleno_compactado.sum()
            cut_vol['material_mejoramiento'] = material_mejoramiento.sum()
            cut_vol['replantillo'] = replantillo.sum()
            cut_vol['hormigon_pobre'] = hormigon_pobre.sum()
            cut_vol['encofrado_recto'] = encofrado.sum()
            cut_vol['encofrado_curvo'] = encofrado.sum() * 0.2
            cut_vol['entibado'] = entibado.sum()
            cut_vol['hormigon_estructura'] = vol_pz.sum()
            cut_vol['peso_acero'] = peso_acero.sum()
            cut_vol['tapas'] = tapas
            cut_vol['estribos'] = estribos_varilla_grada
            cut_vol['acarreo_manual'] = acarreo_manual.sum()
            cut_vol['acarreo_mecanico'] = acarreo_mecanico.sum()
            cut_vol['desalojo'] = desalojo.sum()
            
            cut_vol = cut_vol.to_frame()
            cut_vol.columns = ['CANTIDAD']
            cut_vol['SECTION'] = "POZO DE AVANCE"
            
            return cut_vol
        else:
            return pd.DataFrame([])
    
    def get_tunel_pozo_vol(self):
        df_tunel = self.get_vol_tunel()
        df_pozo_avance = self.get_pozo_avance()
        
        return pd.concat([df_tunel, df_pozo_avance])


class CantidadesTuberiasPozos:
    
    def __init__(self, m_ramales_df, area_trench_properties, parameters_dict):
        self.distancia_desalojo = parameters_dict['distancia_desalojo']
        self.m_ramales_df = m_ramales_df
        self.area_trench_properties = area_trench_properties
    
    def seccion_str2float(self, arr, return_b=False):
        """
        :param arr: array of string mix of circular and rectangular sections
        :return: array of diameter of circular sections and height dimensions of rectangular sections
        """
        # Convert input to pandas Series if it's not already
        arr_series = pd.Series(arr)
        
        # Check if there are any 'x' in the array
        if arr_series.str.contains('x').any():
            # Split base and height when possible
            split_arr = arr_series.str.split('x', expand=True)
            b = split_arr[0].fillna(0).to_numpy().astype(str)
            h = split_arr[1].fillna(0).to_numpy().astype(str)
            
            # Get circular and rectangular indexes
            circular_index = np.char.equal(h, '0').nonzero()[0]
            rectangular_index = np.char.not_equal(h, '0').nonzero()[0]
            
            # Zeros array
            out = np.zeros(shape=len(b), dtype=float)
            
            # Fill zeros array
            out[circular_index] = b[circular_index].astype(float)
            
            if return_b:
                out[rectangular_index] = b[rectangular_index].astype(float)
            else:
                out[rectangular_index] = h[rectangular_index].astype(float)
        else:
            # If there are no 'x', treat all as circular sections
            out = arr_series.astype(float).to_numpy()
        
        return out
    
    def get_length_metodo_constructivo(self):
        
        # Filter for new or to be repaired pipes
        filtro_nuevo = np.logical_and(self.m_ramales_df['Estado'] == 'nuevo', self.m_ramales_df['metodo_constructivo'] != 'existente')
        metodo_constructivo_df = self.m_ramales_df.loc[filtro_nuevo]
        
        metodos = metodo_constructivo_df['metodo_constructivo'].unique()
        secciones = metodo_constructivo_df['Seccion'].unique()
        materiales = metodo_constructivo_df['Material'].unique()
        # Prebuild the result dictionary
        res = {metodo: {seccion: {material: None for material in materiales} for seccion in secciones} for metodo in metodos}
        for metodo in metodos:
            filtro_metodo = metodo_constructivo_df['metodo_constructivo'] == metodo
            metodo_df = metodo_constructivo_df.loc[filtro_metodo]
            for seccion in secciones:
                filtro_seccion = metodo_df['Seccion'] == seccion
                seccion_df = metodo_df.loc[filtro_seccion]
                for material in materiales:
                    filtro_material = seccion_df['Material'] == material
                    material_df = seccion_df.loc[filtro_material]
                    if material_df.empty:
                        continue
                    # Group by 'D_ext' and sum up the 'L' column
                    s = material_df.groupby('D_ext')['L'].sum()
                    # Sort the Series index
                    sorted_index = natsorted(s.index, alg=ns.FLOAT)
                    # Reindex the Series based on the sorted index
                    s_sorted = s.reindex(sorted_index)
                    res[metodo][seccion][material] = s_sorted
        # Filter out entries with None values
        filtered_res = {metodo: {seccion: {material: lengths for material, lengths in materials.items() if lengths is not None} for seccion, materials in secciones.items() if any(lengths is not None for lengths in materials.values())} for metodo, secciones in res.items() if any(any(lengths is not None for lengths in materials.values()) for materials in secciones.values())}
        # Convert to a DataFrame readable by people to Excel
        rows = []
        for metodo, secciones in filtered_res.items():
            for seccion, materiales in secciones.items():
                for material, lengths in materiales.items():
                    for d_ext, length in lengths.items():
                        rows.append([metodo, seccion, material, d_ext, length])
        df = pd.DataFrame(rows, columns=['metodo_constructivo', 'Seccion', 'Material', 'D_ext', 'L'])
        
        self.cantidades_colectores = df
        
        return df
    
    def get_pipe_length_circular(self):
        # Filter for circular pipes
        filtro_circular = self.cantidades_colectores['Seccion'] == 'circular'
        
        # Create a unique index by combining external diameter and material
        indice_circular = self.cantidades_colectores[filtro_circular]['D_ext'] + '-' + self.cantidades_colectores[filtro_circular]['Material']
        
        # Create a new DataFrame with lengths of circular pipes, indexed by the unique combination
        cantidades_pipes = pd.DataFrame(self.cantidades_colectores[filtro_circular]['L'].to_numpy(), index=indice_circular, columns=['CANTIDAD'])
        
        # Group by the unique index and sum the lengths
        cantidades_pipes = cantidades_pipes['CANTIDAD'].groupby(cantidades_pipes.index).sum()
        
        # Convert the series back to a DataFrame
        cantidades_pipes = cantidades_pipes.to_frame()
        
        materiales_list = list(vg.diametro_interno_externo_pypiper.keys())
        section = [f"SUMINISTRO E INSTALACION TUBERIA {material}" for index in cantidades_pipes.index for material in materiales_list if material in index]
        
        # Add a new column 'SECTION' with a constant value
        cantidades_pipes['SECTION'] = section
        
        # Return the final DataFrame
        return cantidades_pipes
    
    def get_pipe_length_rectangular(self):
        # Filter for rectangular sections
        filtro_rectangular = self.m_ramales_df['Seccion'] == 'rectangular'
        
        if len(filtro_rectangular.to_numpy().nonzero()[0]) > 0:
            area_properties_rectangular = self.area_trench_properties[filtro_rectangular]
            # -------------------------------------------------------------------------------------------------------------
            # rect df
            rect_df = self.m_ramales_df[filtro_rectangular].reset_index(drop=True)
            # smaller depth from each trench
            h_min = np.min([rect_df['HF'], rect_df['HI']], axis=0)
            # altura seccion
            h_seccion = self.seccion_str2float(rect_df['D_ext'])
            
            # canal abierto
            filtro_canal_abierto = np.isclose(h_seccion, h_min, atol=0.4)
            filtro_colector_cerrado = np.invert(filtro_canal_abierto)
            espesor_hormigon_pobre = vg.parametros_cantidades['espesor_hormigon pobre']
            
            if len(filtro_canal_abierto[filtro_canal_abierto == True]) > 0:
                # -------------------------------------------------------------------------------------------------------------
                # cantidades de canlal abierto
                df_canal_abierto = rect_df[filtro_canal_abierto].groupby('D_ext')['L'].sum().reset_index()
                
                h_canal_abierto = self.seccion_str2float(df_canal_abierto['D_ext'])
                b_canal_abierto = self.seccion_str2float(df_canal_abierto['D_ext'], return_b=True)
                
                e_pared_canal = vg.parametros_cantidades['espesor_pared_canal_abierto']
                e_fondo_canal = vg.parametros_cantidades['espesor_fondo_canal_abierto']
                cuantia_acero_canal = vg.parametros_cantidades['cuantia_acero_canal_abierto']
                
                area_pared_canal = h_canal_abierto * e_pared_canal * 2
                area_fondo_canal = (b_canal_abierto + (e_pared_canal * 2)) * e_fondo_canal
                
                area_canal = area_fondo_canal + area_pared_canal
                vol_canal = area_canal * df_canal_abierto['L'].to_numpy()
                malla_electrosoldada = vol_canal * cuantia_acero_canal
                
                replantillo_piedra_canal = area_properties_rectangular[filtro_canal_abierto]['base_inferior'] * rect_df['L'][filtro_canal_abierto].to_numpy()
                hormigon_pobre_canal = replantillo_piedra_canal * espesor_hormigon_pobre
                enconfrado_canal = ((h_canal_abierto * 4 + b_canal_abierto) + (e_pared_canal * 2) + (2 * e_fondo_canal)) * df_canal_abierto['L'].to_numpy()
                rejilla = df_canal_abierto['L'].sum().round(0) + 1
                
                cantidades_canal = pd.Series([replantillo_piedra_canal.sum(), hormigon_pobre_canal.sum(), enconfrado_canal.sum(), vol_canal.sum(), malla_electrosoldada.sum(), rejilla], index=['replantillo_canal', 'hormigon_pobre_canal', 'encofrado_canal', 'hormigon_canal', 'malla_electrosoldada_canal', 'rejilla_canal'])
                cantidades_canal = cantidades_canal.to_frame()
                cantidades_canal.columns = ['CANTIDAD']
                cantidades_canal['SECTION'] = 'CANALES ABIERTOS Y REJILLAS'
            
            else:
                cantidades_canal = pd.DataFrame([])
            
            # -------------------------------------------------------------------------------------------------------------
            if len(filtro_colector_cerrado[filtro_colector_cerrado == True]) > 0:
                # cantidades de canlal abierto
                df_colector_cerrado = rect_df[filtro_colector_cerrado].groupby('D_ext')['L'].sum().reset_index()
                
                h_colector_cerrado = self.seccion_str2float(df_colector_cerrado['D_ext'])
                b_colector_cerrado = self.seccion_str2float(df_colector_cerrado['D_ext'], return_b=True)
                
                e_pared_colector_cerrado = vg.parametros_cantidades['espesor_pared_colector_cerrado']
                e_fondo_colector_cerrado = vg.parametros_cantidades['espesor_fondo_colector_cerrado']
                e_tapa_colector_cerrado = vg.parametros_cantidades['espesor_tapa_colector_cerrado']
                cuantia_acero_colector_cerrado = vg.parametros_cantidades['cuantia_acero_colector_cerrado']
                
                area_pared_colector_cerrado = h_colector_cerrado * e_pared_colector_cerrado * 2
                area_fondo_colector_cerrado = (b_colector_cerrado + (2 * e_pared_colector_cerrado)) * e_fondo_colector_cerrado
                area_tapa_colector_cerrado = (b_colector_cerrado + (2 * e_pared_colector_cerrado)) * e_tapa_colector_cerrado
                
                area_colector_cerrado = area_fondo_colector_cerrado + area_pared_colector_cerrado + area_tapa_colector_cerrado
                vol_colector_cerrado = area_colector_cerrado * df_colector_cerrado['L'].to_numpy()
                peso_acero_colector_cerrado = vol_colector_cerrado * cuantia_acero_colector_cerrado
                replantillo_piedra_colector_cerrado = area_properties_rectangular[filtro_colector_cerrado]['base_inferior'] * rect_df['L'][filtro_colector_cerrado].to_numpy()
                hormigon_pobre_colector_cerrado = replantillo_piedra_colector_cerrado * espesor_hormigon_pobre
                enconfrado_colector_cerrado = ((h_colector_cerrado * 4 + 3 * b_colector_cerrado) + (e_pared_colector_cerrado * 2) + (2 * e_fondo_colector_cerrado) + (2 * e_tapa_colector_cerrado)) * df_colector_cerrado['L'].to_numpy()
                
                cantidades_colector_cerrado = pd.Series([replantillo_piedra_colector_cerrado.sum(), hormigon_pobre_colector_cerrado.sum(), enconfrado_colector_cerrado.sum(), vol_colector_cerrado.sum(), peso_acero_colector_cerrado.sum()], index=['replantillo_colector_cerrado', 'hormigon_pobre_colector_cerrado', 'encofrado_colector_cerrado', 'hormigon_colector_cerrado', 'acero_colector_cerrado'])
                cantidades_colector_cerrado = cantidades_colector_cerrado.to_frame()
                cantidades_colector_cerrado.columns = ['CANTIDAD']
                cantidades_colector_cerrado['SECTION'] = 'COLECTOR DE HORMIGON ARMADO'
            
            else:
                cantidades_colector_cerrado = pd.DataFrame([])
            
            # -------------------------------------------------------------------------------------------------------------
            
            return pd.concat([cantidades_canal, cantidades_colector_cerrado])
        
        else:
            return pd.DataFrame([])
    
    def get_pz_class(self):
        
        df_tramo = self.m_ramales_df['Tramo'].str.split('-', expand=True)
        pz_start = df_tramo.iloc[:, 0].str.split('.', expand=True)
        start_pz_index = pz_start.iloc[:, 1] == '0'
        
        start_pz_class = self.m_ramales_df[start_pz_index]['pz_class'].to_numpy()
        start_pz_depth = self.m_ramales_df[start_pz_index]['HI'].to_numpy()
        start_pz_diameter = self.seccion_str2float(self.m_ramales_df[start_pz_index]['D_ext'].to_numpy(), return_b=True)
        
        tramo_pz_class = self.m_ramales_df['pz_class'].to_numpy()
        tramo_pz_depth = self.m_ramales_df['HF'].to_numpy()
        tramo_pz_diameter = self.seccion_str2float(self.m_ramales_df['D_ext'].to_numpy(), return_b=True)
        
        pz_class_array = np.concatenate([start_pz_class, tramo_pz_class])
        depth_array = np.concatenate([start_pz_depth, tramo_pz_depth])
        diameter_array = np.concatenate([start_pz_diameter, tramo_pz_diameter])
        
        filtro_pz_in = pd.Series(pz_class_array).str.contains(r'(pz-b\d+)', regex=True)
        filtro_pz_out = np.invert(filtro_pz_in)
        
        # ---------------------------------------------------------------------------------------------------
        pz_b, filtro_especial_indice = self.classify_pz_depth(depth_array[filtro_pz_in], pz_class_array[filtro_pz_in])
        pz_b = pz_b.to_frame()
        pz_b.columns = ['CANTIDAD']
        pz_b['SECTION'] = 'POZOS DE REVISION'
        
        filtro_out_indice = np.array(filtro_pz_out.to_numpy().nonzero()[0])
        if filtro_out_indice.size > 0:
            filtro_pz_out_indice = np.concatenate([arr for arr in [filtro_out_indice, filtro_especial_indice] if arr.size > 0])
            self.pz_especial = pd.DataFrame([depth_array[filtro_pz_out_indice], diameter_array[filtro_pz_out_indice], pz_class_array[filtro_pz_out_indice]], index=['H', 'D', 'pz']).T
        else:
            self.pz_especial = pd.DataFrame([], index=['H', 'D', 'pz']).T
        
        return pz_b
    
    def classify_pz_depth(self, depths, pz_classes):
        # depth_ranges = {'pz-b1': [(0, 1.75), (1.76, 2.25), (2.26, 2.75), (2.76, 3.25), (3.26, 3.75), (3.76, 4.25), (4.26, 4.75), (4.76, 5.25), (5.26, 5.75)], 'pz-b2': [(0, 2.99), (3.00, 3.49), (3.50, 3.99), (4.00, 4.49), (4.50, 4.99), (5.00, 5.49), (5.50, 6.00)], 'pz-b3': [(0, 2.99), (3.00, 3.49), (3.50, 3.99), (4.00, 4.49), (4.50, 4.99), (5.00, 5.49), (5.50, 6.00)], 'pz-b4': [(0, 2.99), (3.00, 3.49), (3.50, 3.99), (4.00, 4.49), (4.50, 4.99), (5.00, 5.49), (5.50, 6.00)]}
        depth_ranges = {'pz-b1': [(0, 1.75), (1.76, 2.25), (2.26, 2.75), (2.76, 3.25), (3.26, 3.75), (3.76, 4.25), (4.26, 4.75), (4.76, 5.25), (5.26, 7.00)], 'pz-b2': [(0, 2.99), (3.00, 3.49), (3.50, 3.99), (4.00, 4.49), (4.50, 4.99), (5.00, 5.49), (5.50, 7.00)], 'pz-b3': [(0, 2.99), (3.00, 3.49), (3.50, 3.99), (4.00, 4.49), (4.50, 4.99), (5.00, 5.49), (5.50, 7.00)], 'pz-b4': [(0, 2.99), (3.00, 3.49), (3.50, 3.99), (4.00, 4.49), (4.50, 4.99), (5.00, 5.49), (5.50, 7.00)]}
        classified = []
        esp_count = 0
        esp_indices = []
        for i, (depth, pz_class) in enumerate(zip(depths, pz_classes)):
            
            if pz_class.startswith('pz-b'):
                for lower, upper in depth_ranges[pz_class]:
                    cond_close = np.logical_or(np.isclose(upper, depth, atol=0.05), np.isclose(lower, depth, atol=0.05))
                    if lower <= depth < upper or cond_close:
                        classified.append(f"{pz_class}[{lower}-{upper}]")
                        break
                else:
                    esp_count += 1
                    esp_indices.append(i)
            else:
                classified.append(pz_class)
        result = pd.Series(classified).value_counts()
        
        result = result[natsorted(result.index)]
        
        return result, np.array(esp_indices)
    
    def calculate_deptl_ranges_vol(self, trench_depths, areas):
        depth_ranges = self.depth_ranges
        results = {}
        for pos, depth_range in enumerate(depth_ranges):
            empty_arr = np.empty(len(trench_depths))
            
            start, end = depth_range
            condition_in = np.logical_and(trench_depths >= start, trench_depths < end)
            condition_out = np.invert(condition_in)
            
            if start == 0:
                empty_arr[condition_in.nonzero()] = areas[condition_in] * trench_depths[condition_in]
                empty_arr[condition_out.nonzero()] = (end - start) * areas[condition_out]
                results[f'{start}m-{end}m'] = empty_arr
            else:
                previus_filtro = np.invert(trench_depths <= start)
                empty_arr[condition_in.nonzero()] = (trench_depths[condition_in] - start) * areas[condition_in]
                empty_arr[condition_out.nonzero()] = (end - start) * areas[condition_out] * previus_filtro[condition_out]
                results[f'{start}m-{end}m'] = empty_arr
        
        # Create DataFrame with labeled columns for each depth range
        df = pd.DataFrame(results)
        return df
    
    
    
    def asignar_tuberia_accesorios_S2(self, diametros_m, salto):
        """
        Asigna diámetros equivalentes en mm y rango en pulgadas
        """
        # Convertir a mm y asignar el más cercano
        mm = diametros_m * 1000
        diametros_disponibles = np.array([200, 300, 400, 500, 600])
        diametros_mm2inch = {200:8, 300:12, 400: 16, 500:20, 600:24}
        accesorios_schedule_40_kg = {
                # Diámetro (pulg) : {Tipo de accesorio : Peso (kg)}
                8: {
                    "Codo 90° LR (Largo Radio)": 13.15,  # 29 lbs
                    "Unión soldable": 20.41,              # 45 lbs
                },
                12: {
                    "Codo 90° LR (Largo Radio)": 31.75,   # 70 lbs
                    "Unión soldable": 47.63,              # 105 lbs
                },
                16: {
                    "Codo 90° LR (Largo Radio)": 68.04,   # 150 lbs
                    "Unión soldable": 99.79,              # 220 lbs
                },
                20: {
                    "Codo 90° LR (Largo Radio)": 117.94,  # 260 lbs
                    "Unión soldable": 172.37,             # 380 lbs
                },
                24: {
                    "Codo 90° LR (Largo Radio)": 190.51,  # 420 lbs
                    "Unión soldable": 281.23,             # 620 lbs
                }
            }
        
        asignado = pd.Series([diametros_disponibles[np.abs(diametros_disponibles - x).argmin()] for x in mm])
        inches = itemgetter(*asignado)(diametros_mm2inch)
        
        if isinstance(inches, int):
            inches = [inches]
        else:
            inches = list(inches)
            
        peso_codos = [accesorios_schedule_40_kg[_]['Codo 90° LR (Largo Radio)'] * 2 for _ in inches]
        peso_codos_df = pd.DataFrame(np.array([inches, peso_codos]).T, columns=['diametro_pulgadas', 'peso_union'])
        peso_codos_df = peso_codos_df.groupby('diametro_pulgadas')['peso_union'].sum().reset_index()
        peso_codos_series = pd.Series(
            data=peso_codos_df['peso_union'].to_numpy(),
            index=["peso_codo_" + str(int(d)) + '_pulgadas' for d in peso_codos_df ['diametro_pulgadas']]
        )
        
        
        peso_uniones = [ 2 for _ in inches] # no es peso es cantidad de uniones
        peso_uniones_df = pd.DataFrame(np.array([inches, peso_uniones]).T, columns=['diametro_pulgadas', 'peso_union'])
        peso_uniones_df = peso_uniones_df.groupby('diametro_pulgadas')['peso_union'].sum().reset_index()
        peso_uniones_series = pd.Series(
            data=peso_uniones_df['peso_union'].to_numpy(),
            index=["peso_union_" + str(int(d)) + '_pulgadas' for d in peso_uniones_df ['diametro_pulgadas']]
        )
        
        longitud_tuberia_acero = salto * 1.5
        longitud_tuberia_acero_df = pd.DataFrame(np.array([inches, longitud_tuberia_acero]).T, columns=['diametro_pulgadas', 'longitud_tuberia_acero'])
        longitud_tuberia_acero_df  = longitud_tuberia_acero_df .groupby('diametro_pulgadas')['longitud_tuberia_acero'].sum().reset_index()
        longitud_tuberia_acero_series = pd.Series(
            data=longitud_tuberia_acero_df ['longitud_tuberia_acero'].to_numpy(),
            index=["longitud_tuberia_acero_" + str(int(d)) + '_pulgadas' for d in longitud_tuberia_acero_df  ['diametro_pulgadas']]
        )

        return longitud_tuberia_acero_series, peso_uniones_series, peso_codos_series
        
    def get_pz_volumes(self, df, pz_type):
        # get depth
        h = df['H']
        # get diameter
        D = df['D']
        
        # espesores de pared
        e_pared = vg.parametros_cantidades['espesor_pared_pozo']
        e_fondo = vg.parametros_cantidades['espesor_fondo_pozo']
        e_tapa = vg.parametros_cantidades['espesor_tapa_pozo']
        e_hormigon_pobre = vg.parametros_cantidades['espesor_hormigon pobre']
        cuantia_acero = vg.parametros_cantidades['cuantia_acero_pozo']
        seccion_minima_pozo = vg.parametros_cantidades['secion_minima_pozo']
        
        ancho_interno = np.max([D + (0.4 * 2), np.full(len(D), seccion_minima_pozo)], axis=0)
        largo_interno = max(0.5 + e_pared + 1.25, seccion_minima_pozo)
        
        ancho_externo = ancho_interno + (2 * e_pared)
        largo_externo = largo_interno + (2 * e_pared)
        area_externa = ancho_externo * largo_externo
        
        area_interna = ancho_interno * largo_interno
        
        area_paredes = area_externa - area_interna
        area_disipador = ancho_interno * 2
        
        area_seccion = area_disipador + area_paredes
        if pz_type in ['pz-s2']:
            vol_seccion = area_seccion * h
            longitud_tuberia_acero_series, peso_uniones_series, peso_codos_series = self.asignar_tuberia_accesorios_S2(D, h)
        else:
            vol_seccion = area_seccion * h + 0.4
            longitud_tuberia_acero_series, peso_uniones_series, peso_codos_series = pd.Series([]), pd.Series([]), pd.Series([])
        # ---------------------------------------------------------------------------
        vol_tapa = e_tapa * ancho_externo * largo_externo
        vol_fondo = e_fondo * ancho_externo * largo_externo
        vol_entrada = (D + 0.5) * 0.5
        
        vol_pz = vol_seccion + vol_tapa + vol_fondo + vol_entrada * 2
        peso_acero = vol_pz * cuantia_acero
        replantillo = (ancho_externo + 1) * (largo_externo + 1)
        hormigon_pobre = replantillo * e_hormigon_pobre
        encofrado = (area_externa * h) + (area_interna * h) + ((D + 0.4 * 2) * 2 * h) + (area_externa + area_interna * 2) + (e_tapa * (largo_interno * 2 + ancho_externo * 2))
        cut_vol = self.calculate_deptl_ranges_vol(h.to_numpy(), replantillo).sum().replace([np.inf, -np.inf], 0)
        tapas = len(D)
        estribos_varilla_grada = round(np.sum(h / 0.3).sum(), 0)
        acarreo_manual = cut_vol * 0.2 * 20
        acarreo_mecanico = cut_vol * 0.8
        desalojo = cut_vol * self.distancia_desalojo
        relleno_compactado = (replantillo - area_externa) * h * 0.3
        material_mejoramiento = (replantillo - area_externa) * h * 0.7
        entibado = h * (ancho_externo + 1) * 2 + (largo_externo + 1) * 2
        
        cut_vol['relleno_compactado'] = relleno_compactado.sum()
        cut_vol['material_mejoramiento'] = material_mejoramiento.sum()
        cut_vol['replantillo'] = replantillo.sum()
        cut_vol['hormigon_pobre'] = hormigon_pobre.sum()
        cut_vol['encofrado_recto'] = encofrado.sum()
        cut_vol['encofrado_curvo'] = encofrado.sum() * 0.2
        cut_vol['entibado'] = entibado.sum()
        cut_vol['hormigon_estructura'] = vol_pz.sum()
        cut_vol['peso_acero'] = peso_acero.sum()
        cut_vol['tapas'] = tapas
        cut_vol['estribos'] = estribos_varilla_grada
        cut_vol['acarreo_manual'] = acarreo_manual.sum()
        cut_vol['acarreo_mecanico'] = acarreo_mecanico.sum()
        cut_vol['desalojo'] = desalojo.sum()
        
        if pz_type in ['pz-s2']:
            cut_vol['hormigon_ciclopeo'] = np.sum(1.5 * ancho_interno)
            cut_vol['malla_electrosoldada'] = np.sum(ancho_interno * 0.1 * 0.4 * 3 * 80)
        
        cut_vol = pd.concat([cut_vol, longitud_tuberia_acero_series, peso_uniones_series, peso_codos_series])
        
        cut_vol = cut_vol.to_frame()
        cut_vol.columns = ['CANTIDAD']
        
        if pz_type in ['pz-s2']:
            cut_vol['SECTION'] = 'POZO SALTO TIPO S2'
        else:
            cut_vol['SECTION'] = 'POZO SALTO TIPO S1'
        
        return cut_vol
    
    def get_pozo_especial(self, depth_ranges):
        self.depth_ranges = depth_ranges
        df_list = []
        for pz_type, df_pz in self.pz_especial.groupby('pz'):
            if pz_type in ['pz-s1', 'pz-s2']:
                df_list.append(self.get_pz_volumes(df_pz, pz_type))
        
        filtro = np.where(self.pz_especial['pz'].to_numpy() == 'pz-s1', False, True) * np.where(self.pz_especial['pz'].to_numpy() == 'pz-s2', False, True)
        # print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        # print('Pozos a Diseñar')
        # print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        # print(self.pz_especial[filtro])
        # print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        if len(df_list) == 0:
            return pd.DataFrame([])
        else:
            return pd.concat(df_list)


class CantidadesDerrocamiento:
    def __init__(self, parameters_dict):
        
        self.parameters_dict = parameters_dict
        
        # get vector path
        self.vector_path = parameters_dict['vector_path']
        try:
            self.m_ramales_df = gpd.read_file(self.vector_path, engine='pyogrio')
        except:
            self.m_ramales_df = gpd.read_file(self.vector_path)
        
        # filter out only requiered column
        filtro_dict = parameters_dict.get('filtro')
        if filtro_dict:
            filtro_column = filtro_dict.get('column')
            filtro_value = filtro_dict.get('value')
            
            if filtro_column and filtro_value:
                # Convert filtro_value to a list if it's a string of comma-separated values
                if isinstance(filtro_value, str):
                    filtro_value = [x.strip() for x in filtro_value.split(',')]
                
                # Apply the filter using isin to handle lists of values
                filtro = self.m_ramales_df[filtro_column].isin(filtro_value)
                self.m_ramales_df = self.m_ramales_df.loc[filtro]
        
        # volumen de tramos nuevos
        filtro_nuevo = self.m_ramales_df['Estado'] == 'nuevo'
        self.m_ramales_df = self.m_ramales_df.loc[filtro_nuevo]
        
        # tramos
        self.m_ramales_df = self.m_ramales_df.reset_index(drop=True)
    
    def seccion_str2float(self, arr, return_b=False):
        """
    :param arr: array of string mix of circular and rectangular sections
    :return: array of diameter of circular sections and height dimensions of rectangular sections
        """
        # Convert input to pandas Series if it's not already
        arr_series = pd.Series(arr)
        
        # Check if there are any 'x' in the array
        if arr_series.str.contains('x').any():
            # Split base and height when possible
            split_arr = arr_series.str.split('x', expand=True)
            b = split_arr[0].fillna(0).to_numpy().astype(str)
            h = split_arr[1].fillna(0).to_numpy().astype(str)
            
            # Get circular and rectangular indexes
            circular_index = np.char.equal(h, '0').nonzero()[0]
            rectangular_index = np.char.not_equal(h, '0').nonzero()[0]
            
            # Zeros array
            out = np.zeros(shape=len(b), dtype=float)
            
            # Fill zeros array
            out[circular_index] = b[circular_index].astype(float)
            
            if return_b:
                out[rectangular_index] = b[rectangular_index].astype(float)
            else:
                out[rectangular_index] = h[rectangular_index].astype(float)
        else:
            # If there are no 'x', treat all as circular sections
            out = arr_series.astype(float).to_numpy()
        
        return out
    
    def get_derrocamiento_vol(self):
        filtro_circular = self.m_ramales_df['Seccion'] == 'circular'
        filtro_rectangular = self.m_ramales_df['Seccion'] == 'rectangular'
        
        vol_circular = np.pi * np.sqrt(self.m_ramales_df[filtro_circular]['D_ext'].astype(float).to_numpy() / 2.0) * self.m_ramales_df[filtro_circular]['L'].to_numpy()
        vol_rectangular = self.seccion_str2float(self.m_ramales_df[filtro_rectangular]['D_ext']) * self.seccion_str2float(self.m_ramales_df[filtro_rectangular]['D_ext'], return_b=True)
        vol = vol_circular.sum() + vol_rectangular.sum()
        
        s = pd.Series([vol * 0.01, vol * 0.01, vol * 0.01, vol * 0.01], index=['derrocamiento_hormigon_armado', 'derrocamiento_pozo_hormigon_simple', 'derrocamiento_manposteria_ladrillo', 'derrocamiento_manposteria_bloque'])
        
        s = s.to_frame()
        s.columns = ['CANTIDAD']
        s['SECTION'] = "DEROCAMIENTO"
        return s


class CantidadesAdicionales:
    def __init__(self, parameters_dict, cantidad_pozos):
        
        self.parameters_dict = parameters_dict
        self.cantidad_pozos = cantidad_pozos
        # get vector path
        self.vector_path = parameters_dict['vector_path']
        try:
            self.m_ramales_df = gpd.read_file(self.vector_path, engine='pyogrio')
        except:
            self.m_ramales_df = gpd.read_file(self.vector_path)
        
        # filter out only requiered column
        filtro_dict = parameters_dict.get('filtro')
        if filtro_dict:
            filtro_column = filtro_dict.get('column')
            filtro_value = filtro_dict.get('value')
            
            if filtro_column and filtro_value:
                # Convert filtro_value to a list if it's a string of comma-separated values
                if isinstance(filtro_value, str):
                    filtro_value = [x.strip() for x in filtro_value.split(',')]
                
                # Apply the filter using isin to handle lists of values
                filtro = self.m_ramales_df[filtro_column].isin(filtro_value)
                self.m_ramales_df = self.m_ramales_df.loc[filtro]
        
        # volumen de tramos nuevos
        filtro_nuevo = self.m_ramales_df['Estado'] == 'nuevo'
        self.m_ramales_df = self.m_ramales_df.loc[filtro_nuevo]
        
        # tramos
        self.m_ramales_df = self.m_ramales_df.reset_index(drop=True)
    
    def seccion_str2float(self, arr, return_b=False):
        """
        :param arr: array of string mix of circular and rectangular sections
        :return: array of diameter of circular sections and height dimensions of rectangular sections
        """
        # Convert input to pandas Series if it's not already
        arr_series = pd.Series(arr)
        
        # Check if there are any 'x' in the array
        if arr_series.str.contains('x').any():
            # Split base and height when possible
            split_arr = arr_series.str.split('x', expand=True)
            b = split_arr[0].fillna(0).to_numpy().astype(str)
            h = split_arr[1].fillna(0).to_numpy().astype(str)
            
            # Get circular and rectangular indexes
            circular_index = np.char.equal(h, '0').nonzero()[0]
            rectangular_index = np.char.not_equal(h, '0').nonzero()[0]
            
            # Zeros array
            out = np.zeros(shape=len(b), dtype=float)
            
            # Fill zeros array
            out[circular_index] = b[circular_index].astype(float)
            
            if return_b:
                out[rectangular_index] = b[rectangular_index].astype(float)
            else:
                out[rectangular_index] = h[rectangular_index].astype(float)
        else:
            # If there are no 'x', treat all as circular sections
            out = arr_series.astype(float).to_numpy()
        
        return out
    
    def agrupar_diametros_longitudes(self, diametros, longitudes):
        # Crear un DataFrame con los diámetros y longitudes
        df = pd.DataFrame({'diametro': diametros * 1000,  # Convertir a mm
                           'longitud': longitudes})
        # Definir los rangos y sus códigos
        rangos = [(200, 400, 'CTV_200_400'), (450, 750, 'CTV_450_750'), (800, 1500, 'CTV_800_1500')]
        # Crear columna para código
        df['codigo'] = 'Fuera de rango'
        # Asignar códigos basados en los rangos
        for min_d, max_d, codigo in rangos:
            mascara = (df['diametro'] >= min_d) & (df['diametro'] <= max_d)
            df.loc[mascara, 'codigo'] = codigo
        # Agrupar por código y calcular la suma de longitudes
        resultado = df.groupby('codigo')['longitud'].sum()
        return resultado
    
    def get_CCTV(self):
        
        # diametros = self.seccion_str2float(self.m_ramales_df['D_ext'].to_numpy(), return_b=True)
        longitudes = self.m_ramales_df['L'].to_numpy()
        # other_quantities = self.agrupar_diametros_longitudes(diametros, longitudes)
        other_quantities = pd.Series({'CTV': longitudes.sum()})
        
        other_quantities = other_quantities.to_frame()
        other_quantities.columns = ['CANTIDAD']
        other_quantities['SECTION'] = 'ADICIONALES'
        
        return other_quantities
    
    def get_socio_ambiental(self, numero_domiciliarias):
        L = self.m_ramales_df['L'].to_numpy()
        D = self.seccion_str2float(self.m_ramales_df['D_ext'].to_numpy()) + 0.5
        H = self.m_ramales_df['HF'].to_numpy()
        vol = np.sum(L * D * H)
        ancho_plastico = 2
        largo_plastico = vol / (ancho_plastico * 1.2)
        L = L.sum()
        
        area_plastico = ancho_plastico * largo_plastico * 0.5
        rotulos = int(L / 450) * 4
        conos = int(L / 50) * 2
        cierre_transito = int(L / 90)
        cierre_via = int(L / 200)
        barricada = int(L / 50)
        barricada_plastica = int(L / 100)
        paso_peatonal = int(L / 40)
        cerramiento_provisional_tool = int(L / 800) * 2 * 20
        cubierta_provisional = int(L / 1000) * 2 * 20
        control_polvo = int(vol.sum() * 0.05)
        
        # Crear la serie
        datos = {'area_plastico': area_plastico, 'rotulos': rotulos, 'conos': conos, 'cierre_transito': cierre_transito, 'cierre_via': cierre_via, 'barricada': barricada, 'barricada_plastica': barricada_plastica, 'paso_peatonal': paso_peatonal, 'cerramiento_provisional_tool': cerramiento_provisional_tool, 'cubierta_provisional': cubierta_provisional, 'control_polvo': control_polvo, 'ficha_catastral': numero_domiciliarias,  # borrar
                 }
        serie = pd.Series(datos)
        
        socio_ambiental_quantities = serie.to_frame()
        socio_ambiental_quantities.columns = ['CANTIDAD']
        socio_ambiental_quantities['SECTION'] = 'PLAN DE MANEJO SOCIOAMBIENTAL'
        
        return socio_ambiental_quantities
    
    def get_catastro(self):
        L = self.m_ramales_df['L'].to_numpy().sum() / 1000
        
        # Crear la serie
        datos = {'catastro': L, 'objeto_as_built': self.cantidad_pozos, }
        serie = pd.Series(datos)
        
        adicional_quantities = serie.to_frame()
        adicional_quantities.columns = ['CANTIDAD']
        adicional_quantities['SECTION'] = 'ADICIONALES'
        
        return adicional_quantities


class CantidadesDomiciliariasSumideros:
    def __init__(self, parameters_dict, tipo, fase):
        
        self.fase = fase
        self.parameters_dict = parameters_dict
        self.tipo = tipo
        
        # get vector path
        self.vector_path = parameters_dict['vector_path']
        try:
            self.m_ramales_df = gpd.read_file(self.vector_path, engine='pyogrio')
        except:
            self.m_ramales_df = gpd.read_file(self.vector_path)
        
        # filter out only requiered column
        filtro_dict = parameters_dict.get('filtro')
        if filtro_dict:
            filtro_column = filtro_dict.get('column')
            filtro_value = filtro_dict.get('value')
            
            if filtro_column and filtro_value:
                # Convert filtro_value to a list if it's a string of comma-separated values
                if isinstance(filtro_value, str):
                    filtro_value = [x.strip() for x in filtro_value.split(',')]
                
                # Apply the filter using isin to handle lists of values
                filtro = self.m_ramales_df[filtro_column].isin(filtro_value)
                self.m_ramales_df = self.m_ramales_df.loc[filtro]
        
        # volumen de tramos nuevos
        filtro_nuevo = self.m_ramales_df['Estado'] == 'nuevo'
        self.m_ramales_df = self.m_ramales_df.loc[filtro_nuevo]
        
        # tramos
        self.m_ramales_df = self.m_ramales_df.reset_index(drop=True)
    
    def asignar_tramos_poligonos_domiciliarias(self, poligonos_path, columnas_datos=[], distancia_maxima=12):
        """
        Asigna polígonos a tramos usando distance_matrix de scipy con restricción de distancia
        y evitando duplicados
        """
        
        poligonos_gdf = gpd.read_file(poligonos_path, engine='pyogrio')
        
        # Step 2: load points form m_ramales
        df_lines = self.m_ramales_df.copy()
        inserted_points_in_linestring = df_lines.geometry.get_coordinates(index_parts=True)
        
        # Prepare groups and coordinates for distance computation
        groups = np.array(inserted_points_in_linestring.index.get_level_values(0)).astype(np.int64)
        coords = inserted_points_in_linestring.to_numpy().copy()
        interval = 2.0
        
        # Insert additional points into linestrings
        all_points, actual_count_of_points = nb_insert_points_in_linestring(coords, groups, interval)
        
        # Create new linestring geometries
        new_geometry = [LineString(points[:count, :]) for points, count in zip(all_points, actual_count_of_points)]
        
        # Create GeoDataFrame with new linestrings
        new_df_lines = gpd.GeoDataFrame(geometry=new_geometry)
        new_coords_df = new_df_lines.geometry.get_coordinates(index_parts=True)
        pypiper_points = new_coords_df.to_numpy()
        pypiper_index = np.array(new_coords_df.index.get_level_values(0))
        new_df_lines['Pozo'] = df_lines['Pozo']
        ref_index = new_df_lines.loc[pypiper_index, 'Pozo']
        
        # Get points from polygon
        polygon_coords_df = poligonos_gdf.get_coordinates(index_parts=True)
        polygon_index = polygon_coords_df.index.get_level_values(0).to_numpy()
        polygon_points = polygon_coords_df.to_numpy()
        
        # Calculate distances
        dist = distance.cdist(polygon_points.copy(), pypiper_points.copy())
        minima_distancia_a_tramo = np.zeros(len(poligonos_gdf))  # Initialize with correct length
        dist_arg_min = np.empty(shape=len(poligonos_gdf), dtype=np.int64)
        
        # Modified loop to ensure we process all polygons
        for grupo in range(len(poligonos_gdf)):
            filter_index = polygon_index == grupo
            if np.any(filter_index):
                numeric_index = filter_index.nonzero()[0]
                dist_group = dist[numeric_index, :]
                
                min_distance_list = np.min(dist_group, axis=1)
                min_index_list = np.argmin(dist_group, axis=1)
                
                min_index = min_index_list[np.argmin(min_distance_list)]
                dist_arg_min[grupo] = min_index
                minima_distancia_a_tramo[grupo] = np.min(min_distance_list)
            else:
                # Handle case where no points are found for the polygon
                dist_arg_min[grupo] = 0
                minima_distancia_a_tramo[grupo] = float('inf')
        
        # Assign results to GeoDataFrame
        id_pozo = ref_index.to_numpy()[dist_arg_min]
        ramal_pozo = pd.Series(id_pozo).str.split('.', expand=True)
        ramal, pozo = ramal_pozo[0], ramal_pozo[1]
        
        poligonos_gdf['Distancia'] = minima_distancia_a_tramo
        poligonos_gdf['Ramal'] = ramal
        poligonos_gdf['Pozo'] = id_pozo
        poligonos_gdf['Fase'] = self.fase
        
        # Calculate coverage
        map_h2pz = poligonos_gdf['Pozo'].map(dict(zip(self.m_ramales_df['Pozo'], self.m_ramales_df[['HF', 'HI']].min(axis=1)))).to_numpy()
        map_D2pz = poligonos_gdf['Pozo'].map(dict(zip(self.m_ramales_df['Pozo'], self.m_ramales_df['D_ext']))).to_numpy()
        poligonos_gdf['Cobertura'] = map_h2pz - self.seccion_str2float(map_D2pz)
        
        # Apply distance filter
        filtro_distancia = (poligonos_gdf['Distancia'] < distancia_maxima)
        filtro_dict = self.parameters_dict.get('filtro')
        # if filtro_dict:
        #     filtro_column = filtro_dict.get('column')
        #     filtro_value = filtro_dict.get('value')
        #
        #     poligonos_gdf[out_cols][filtro_distancia].to_file(f'domiciliarias_{filtro_column}_{filtro_value}_{self.tipo}.gpkg')
        # else:
        #     poligonos_gdf[columnas_datos + ['geometry']][filtro_distancia].to_file(f'domiciliarias_{self.tipo}.gpkg')
        
        # Prepare output columns
        out_cols = columnas_datos + ['Pozo', 'Ramal', 'Cobertura', 'Distancia', 'Fase', 'geometry']
        
        return filtro_distancia.sum(), poligonos_gdf[out_cols][filtro_distancia]
    
    def seccion_str2float(self, arr, return_b=False):
        """
    :param arr: array of string mix of circular and rectangular sections
    :return: array of diameter of circular sections and height dimensions of rectangular sections
        """
        # Convert input to pandas Series if it's not already
        arr_series = pd.Series(arr)
        
        # Check if there are any 'x' in the array
        if arr_series.str.contains('x').any():
            # Split base and height when possible
            split_arr = arr_series.str.split('x', expand=True)
            b = split_arr[0].fillna(0).to_numpy().astype(str)
            h = split_arr[1].fillna(0).to_numpy().astype(str)
            
            # Get circular and rectangular indexes
            circular_index = np.char.equal(h, '0').nonzero()[0]
            rectangular_index = np.char.not_equal(h, '0').nonzero()[0]
            
            # Zeros array
            out = np.zeros(shape=len(b), dtype=float)
            
            # Fill zeros array
            out[circular_index] = b[circular_index].astype(float)
            
            if return_b:
                out[rectangular_index] = b[rectangular_index].astype(float)
            else:
                out[rectangular_index] = h[rectangular_index].astype(float)
        else:
            # If there are no 'x', treat all as circular sections
            out = arr_series.astype(float).to_numpy()
        
        return out
    
    def get_domiciliarias(self):
        
        if self.parameters_dict.get('domiciliarias').get('domiciliarias_vector_path'):
            # cantidad_till, gdf_polygonos = self.asignar_tramos_poligonos_domiciliarias(self.parameters_dict.get('domiciliarias').get('domiciliarias_vector_path'), columnas_datos=['Denominaci', 'Predio', 'ClavCatast'])
            cantidad_till, gdf_polygonos = self.asignar_tramos_poligonos_domiciliarias(self.parameters_dict.get('domiciliarias').get('domiciliarias_vector_path'))
            filtro_profunidad_til = gdf_polygonos['Cobertura'] >= 0.7
            filtro_profunidad_revision = ~filtro_profunidad_til
            cantidad_til = filtro_profunidad_til.sum()
            cantidad_revision = filtro_profunidad_revision.sum()
            longitud_total = gdf_polygonos['Distancia'].sum().item() * 1.2
        else:
            #ToDo: hay que revisar esto,
            cantidad_till = self.parameters_dict.get('domiciliarias').get('numero_domiciliarias')
            longitud_tuberia = self.parameters_dict['domiciliarias']['longitud_tuberia']
            cantidad_revision = 1
            # longitud_til = cantidad_til * longitud_tuberia
            # longitud_revision = 0
        
        ancho = self.parameters_dict['domiciliarias']['ancho_zanja']
        profunidad = self.parameters_dict['domiciliarias']['profundidad_zanja']
        porcentaje_acarreo_manual = self.parameters_dict['domiciliarias']['porcentaje_acarreo_manual']
        porcentaje_acarreo_mecanica = self.parameters_dict['domiciliarias']['porcentaje_acarreo_mecanica']
        distancia_acarreo = self.parameters_dict['distancia_desalojo']
        esponjamiento = self.parameters_dict['porcentaje_esponjamiento']
        distancia_acarreo_manual = self.parameters_dict['distancia_acarreo_manual']
        porcentaje_relleno_compactado = self.parameters_dict['domiciliarias']['porcentaje_relleno_compactado']
        porcentaje_material_mejoramiento = self.parameters_dict['domiciliarias']['porcentaje_material_mejoramiento']
        
        if cantidad_till:
            # pozo til para pozo con cobertura mayor a
            if longitud_total > 0:
                vol = profunidad * ancho * longitud_total
                
                # print('pozo_til_plastico', max(cantidad_til, 1))
                # print('pozo_caja_revision', max(cantidad_revision, 1))
                # Crear la serie
                datos = {'0.0-2.75m-mano-sin clasificar': vol * 0.5,
                         '0.0-2.75m-mano-conglomerado': vol * 0.3,
                         '0.0-2.75m-mano-roca': vol * 0.2,
                         'relleno_compactado': vol * porcentaje_relleno_compactado,
                         'mejoramiento_vol': vol * porcentaje_material_mejoramiento,
                         '0.160-PVC': longitud_total,
                         'pozo_til_plastico': max(cantidad_til, 1),
                         'pozo_caja_revision': max(cantidad_revision, 1),
                         'adaptador_pvc': cantidad_til + cantidad_revision,
                         'acarreo_manual': vol * porcentaje_acarreo_manual * distancia_acarreo_manual,
                         'acarreo_mecanico': vol * porcentaje_acarreo_mecanica,
                         'desalojo' : vol * (1 + esponjamiento) * distancia_acarreo, 'catastro_domiciliaria': cantidad_til + cantidad_revision}
                serie = pd.Series(datos)
                adicional_quantities = serie.to_frame()
                adicional_quantities.columns = ['CANTIDAD']
                adicional_quantities['SECTION'] = 'DOMICILIARIAS'
                return adicional_quantities
            else:
                return pd.Series()
        else:
            return pd.Series()
    
    def get_sumideros(self):
        distancia = self.parameters_dict['sumideros']['distancia_entre_sumideros']
        if distancia:
            longitud = self.m_ramales_df['L'].sum()
            cantidad = int(longitud / distancia) * 2
            
            if cantidad > 0:
                longitud_tuberia = self.parameters_dict['sumideros']['longitud_tuberia']
                ancho = self.parameters_dict['sumideros']['ancho_zanja']
                profunidad = self.parameters_dict['sumideros']['profundidad_zanja']
                porcentaje_acarreo_manual = self.parameters_dict['sumideros']['porcentaje_acarreo_manual']
                porcentaje_acarreo_mecanica = self.parameters_dict['sumideros']['porcentaje_acarreo_mecanica']
                distancia_acarreo = self.parameters_dict['distancia_desalojo']
                esponjamiento = self.parameters_dict['porcentaje_esponjamiento']
                distancia_acarreo_manual = self.parameters_dict['distancia_acarreo_manual']
                porcentaje_relleno_compactado = self.parameters_dict['sumideros']['porcentaje_relleno_compactado']
                porcentaje_material_mejoramiento = self.parameters_dict['sumideros']['porcentaje_material_mejoramiento']
                vol = profunidad * ancho * longitud_tuberia * cantidad
                
                # Crear la serie
                datos = {'rejillas': cantidad, 'pozo_sumidero': cantidad, '0.200-PVC': cantidad * longitud_tuberia, '0.0-2.75m-mano-sin clasificar': vol * 0.5, '0.0-2.75m-mano-conglomerado': vol * 0.3, '0.0-2.75m-mano-roca': vol * 0.2, 'relleno_compactado': vol * porcentaje_relleno_compactado, 'mejoramiento_vol': vol * porcentaje_material_mejoramiento, 'acarreo_manual': vol * porcentaje_acarreo_manual * distancia_acarreo_manual, 'acarreo_mecanico': vol * porcentaje_acarreo_mecanica, 'desalojo': vol * (1 + esponjamiento) * distancia_acarreo, }
                serie = pd.Series(datos)
                
                adicional_quantities = serie.to_frame()
                adicional_quantities.columns = ['CANTIDAD']
                adicional_quantities['SECTION'] = 'SUMIDEROS'
                
                return adicional_quantities
            
            else:
                return pd.Series()
        
        else:
            return pd.Series()


class CantidadesReparacionesAguaPotable_DesvioAlcantarillado:
    def __init__(self, parameters_dict, numero_domiciliarias):
        
        self.parameters_dict = parameters_dict
        self.numero_domiciliarias = numero_domiciliarias
        
        # get vector path
        self.vector_path = parameters_dict['vector_path']
        try:
            self.m_ramales_df = gpd.read_file(self.vector_path, engine='pyogrio')
        except:
            self.m_ramales_df = gpd.read_file(self.vector_path)
        
        # filter out only requiered column
        filtro_dict = parameters_dict.get('filtro')
        if filtro_dict:
            filtro_column = filtro_dict.get('column')
            filtro_value = filtro_dict.get('value')
            
            if filtro_column and filtro_value:
                # Convert filtro_value to a list if it's a string of comma-separated values
                if isinstance(filtro_value, str):
                    filtro_value = [x.strip() for x in filtro_value.split(',')]
                
                # Apply the filter using isin to handle lists of values
                filtro = self.m_ramales_df[filtro_column].isin(filtro_value)
                self.m_ramales_df = self.m_ramales_df.loc[filtro]
        
        # volumen de tramos nuevos
        filtro_nuevo = self.m_ramales_df['Estado'] == 'nuevo'
        self.m_ramales_df = self.m_ramales_df.loc[filtro_nuevo]
        
        # tramos
        self.m_ramales_df = self.m_ramales_df.reset_index(drop=True)
    
    def convert_string_with_arrays(self, s):
        # Replace 'array(' with 'np.array('
        s = s.replace('array(', 'np.array(')
        
        # Create a dictionary of safe globals that includes numpy as np
        safe_dict = {'np': np}
        
        # Evaluate the string with the safe dictionary
        return eval(s, safe_dict)
    
    def get_repair_pipes_length(self):
        
        # check if the file has interferencias
        if 'Interferencia' in self.m_ramales_df.columns:
            filtro_interferencias = self.m_ramales_df['Interferencia'] != 'None'
            cantidad_de_interferencias = []
            for interferencia in self.m_ramales_df.loc[filtro_interferencias]['Interferencia']:
                cantidad_de_interferencias.append(self.convert_string_with_arrays(interferencia)['D'].size)
            
            cantidad_de_interferencias = np.max([np.sum(cantidad_de_interferencias), 1])
            return cantidad_de_interferencias * self.parameters_dict['interferencias_agua_potable']['longitud_tuberia']
        else:
            return self.m_ramales_df['L'].sum() * self.parameters_dict['interferencias_agua_potable']['longitud_tuberia']
    
    def get_quantities_reparacion_tuberia_acero(self):
        """
        Calcula cantidades para reparación de tubería de acero
        
        Parámetros:
        -----------
        L_total : float
            Longitud total de tubería a reparar
        numero_domiciliarias : int
            Número de conexiones domiciliarias
        
        Retorna:
        --------
        pd.DataFrame
            Cantidades para cada rubro
        """
        # Pesos de tubería (kg/m) para diferentes diámetros
        peso_tuberia = {'dos_pulg'   : 7.48,  # 2 pulgadas schedule 40
                        'tres_pulg'  : 11.29,  # 3 pulgadas schedule 40
                        'cuatro_pulg': 16.07,  # 4 pulgadas schedule 40
                        'seis_pulg'  : 28.26  # 6 pulgadas schedule 40
                        }
        
        L_total = self.get_repair_pipes_length()
        
        # Distribuir longitud total entre diferentes diámetros
        L_distribucion = {'dos_pulg'   : L_total * 0.4,  # 30% para 2 pulgadas
                          'tres_pulg'  : L_total * 0.3,  # 30% para 3 pulgadas
                          'cuatro_pulg': L_total * 0.2,  # 20% para 4 pulgadas
                          'seis_pulg'  : L_total * 0.1  # 20% para 6 pulgadas
                          }
        
        # Calcular cantidades
        datos = {'tuberia_acero_2_pulg': L_distribucion['dos_pulg'], 'tuberia_acero_3_pulg': L_distribucion['tres_pulg'], 'tuberia_acero_4_pulg': L_distribucion['cuatro_pulg'], 'tuberia_acero_6_pulg': L_distribucion['seis_pulg'],
                 
                 'union_mecanica_2_pulg': int(L_distribucion['dos_pulg'] / 6) + 1, 'union_mecanica_3_pulg': int(L_distribucion['tres_pulg'] / 6) + 1, 'union_mecanica_4_pulg': int(L_distribucion['cuatro_pulg'] / 6) + 1, 'union_mecanica_6_pulg': int(L_distribucion['seis_pulg'] / 6) + 1,
                 
                 'accesorio_acero_2_pulg': L_distribucion['dos_pulg'] * peso_tuberia['dos_pulg'] * 0.15, 'accesorio_acero_3_pulg': L_distribucion['tres_pulg'] * peso_tuberia['tres_pulg'] * 0.15, 'accesorio_acero_4_pulg': L_distribucion['cuatro_pulg'] * peso_tuberia['cuatro_pulg'] * 0.15, 'accesorio_acero_6_pulg': L_distribucion['seis_pulg'] * peso_tuberia['seis_pulg'] * 0.15,
                 
                 'suelda_tuberia_grande': (L_distribucion['cuatro_pulg'] / 6 * (2 * np.pi * 0.0508) +  # 4 inch = 0.1016m diameter, radius = 0.0508m
                                           L_distribucion['seis_pulg'] / 6 * (2 * np.pi * 0.0762)  # 6 inch = 0.1524m diameter, radius = 0.0762m
                                           ), 'suelda_tuberia_pequena': (L_distribucion['dos_pulg'] / 6 * (2 * np.pi * 0.0254) +  # 2 inch = 0.0508m diameter, radius = 0.0254m
                                                                         L_distribucion['tres_pulg'] / 6 * (2 * np.pi * 0.0381)  # 3 inch = 0.0762m diameter, radius = 0.0381m
                                                                         ),
                 
                 'corte_tuberia_acero': (L_distribucion['dos_pulg'] / 6 * (2 * np.pi * 0.0254) +  # 2 inch
                                         L_distribucion['tres_pulg'] / 6 * (2 * np.pi * 0.0381) +  # 3 inch
                                         L_distribucion['cuatro_pulg'] / 6 * (2 * np.pi * 0.0508) +  # 4 inch
                                         L_distribucion['seis_pulg'] / 6 * (2 * np.pi * 0.0762)  # 6 inch
                                         ) * 2}
        
        # Crear la serie
        serie = pd.Series(datos)
        
        adicional_quantities = serie.to_frame()
        adicional_quantities.columns = ['CANTIDAD']
        adicional_quantities['SECTION'] = 'REPARACION Y REPOSICION TUBERIA DE ACERO DE AGUA POTABLE'
        
        return adicional_quantities
    
    def get_quantities_reparacion_domiciliarias(self):
        # Número total de domiciliarias a reparar
        num_domiciliarias = int(self.numero_domiciliarias * self.parameters_dict['interferencias_agua_potable']['porcentaje_reparacion_domiciliaria'])
        
        # Distribución de diámetros de tubería principal
        distribucion_diametros = {'63mm' : 0.4,  # 40% conexiones a tubería de 63mm
                                  '90mm' : 0.3,  # 30% conexiones a tubería de 90mm
                                  '110mm': 0.2,  # 20% conexiones a tubería de 110mm
                                  '160mm': 0.1  # 10% conexiones a tubería de 160mm
                                  }
        
        # Longitud promedio por domiciliaria
        L_promedio = self.parameters_dict['interferencias_agua_potable']['longitud_tuberia_domiciliaria']
        
        # Calcular cantidades
        datos = {'tuberia_cobre_flexible': num_domiciliarias * L_promedio,  # '02.024.4409'
                 'union_dos_partes'      : num_domiciliarias * 2,  # '02.024.4572'
                 'toma_incorporacion'    : num_domiciliarias,  # '02.024.4740'
                 'valvula_compuerta'     : num_domiciliarias,  # '02.024.4742'
                 'codo_90_acero_inox'    : num_domiciliarias * 2,  # '02.024.4753'
                 'neplo_acero_inox'      : num_domiciliarias * 2,  # '02.024.4755'
                 'union_universal'       : num_domiciliarias,  # '02.024.4782'
                 
                 # Collares según distribución de diámetros
                 'collar_63mm'           : int(num_domiciliarias * distribucion_diametros['63mm']),  # '02.024.4715'
                 'collar_90mm'           : int(num_domiciliarias * distribucion_diametros['90mm']),  # '02.024.4718'
                 'collar_110mm'          : int(num_domiciliarias * distribucion_diametros['110mm']),  # '02.024.4721'
                 'collar_160mm'          : int(num_domiciliarias * distribucion_diametros['160mm'])  # '02.024.4724'
                 }
        
        # Crear la serie
        serie = pd.Series(datos)
        
        # Crear el DataFrame
        adicional_quantities = serie.to_frame()
        adicional_quantities.columns = ['CANTIDAD']
        adicional_quantities['SECTION'] = 'REPARACION Y REPOSICION DOMICILIARIAS DE AGUA POTABLE'
        
        return adicional_quantities
    
    def get_quantities_tuberias_desvio(self):
        # Obtener el DataFrame con diámetros y longitudes
        df_tuberias = self.m_ramales_df.groupby('D_ext')['L'].sum().to_frame(name='L').reset_index()
        
        # Procesar los diámetros
        split_diameter = df_tuberias['D_ext'].str.split('x', expand=True)
        if isinstance(split_diameter, pd.DataFrame):
            if split_diameter.shape[1] == 1:
                df_tuberias['diametro'] = np.where(split_diameter[0].notna(), split_diameter[0], df_tuberias['D_ext'])
            else:
                df_tuberias['diametro'] = np.where(split_diameter[1].notna(), split_diameter[1], df_tuberias['D_ext'])
        else:
            df_tuberias['diametro'] = split_diameter
        
        # Convertir diámetros a float y de metros a milímetros
        df_tuberias['diametro'] = df_tuberias['diametro'].astype(float) * 1000  # Convertir a mm
        
        # Vectorizar la asignación de tuberías de desvío
        diametros_desvio = np.array([200, 300, 600, 900])  # Estos ya están en mm
        diferencias = np.abs(df_tuberias['diametro'].values[:, np.newaxis] - diametros_desvio)
        idx_cercanos = np.argmin(diferencias, axis=1)
        
        # Corregir la concatenación de strings usando pandas
        df_tuberias['TUBERIA_DESVIO'] = pd.Series(diametros_desvio[idx_cercanos]).apply(lambda x: f'tuberia_desvio_{x}mm')
        
        # Calcular longitudes de desvío
        porcentaje_desvio = self.parameters_dict['tuberia_desvio_temporal']['porcentaje_longitud_desvio']
        numero_de_usos = self.parameters_dict['tuberia_desvio_temporal']['numero_de_usos']
        df_tuberias['LONGITUD_DESVIO'] = df_tuberias['L'] * porcentaje_desvio / numero_de_usos
        
        # Agrupar y redondear a múltiplos de 6
        datos = df_tuberias.groupby('TUBERIA_DESVIO')['LONGITUD_DESVIO'].sum()
        datos = ((datos / 6).round(0) * 6).astype(int)
        
        # Crear serie final
        serie = {f'tuberia_desvio_{d}mm': datos.get(f'tuberia_desvio_{d}mm', 0) for d in diametros_desvio}
        
        # Crear el DataFrame final
        adicional_quantities = pd.Series(serie).to_frame()
        adicional_quantities.columns = ['CANTIDAD']
        adicional_quantities['SECTION'] = 'TUBERIAS DE DESVIO TEMPORAL'
        
        return adicional_quantities


# --------------------------------------------------------------------------------------------------------------------------------------------
class RamalesCronogramaNetworkAnalysis:
    """Class to perform network analysis on data related to drainage systems."""
    
    def __init__(self):
        """
        Initialize the NetworkAnalysis class.
        """
    
    def get_network(self, df):
        """
        Create a directed graph from a DataFrame by parsing connections between sources and targets with attributes.

        Args:
            df (DataFrame): DataFrame where each row represents a connection with 'Ramal' data.

        Returns:
            G (DiGraph): A directed graph representing connections.
        """
        
        # Initialize a directed graph
        G = nx.DiGraph()
        
        # Iterate over the DataFrame rows
        for connection, row in df.iterrows():
            # Skip rows where the connection string is empty
            if connection == "":
                continue
            
            # Parse the source and target nodes from the connection string
            source, target = connection.split("-")
            
            # Add nodes and edge to the graph with 'Ramal' as an edge attribute
            G.add_node(source)
            G.add_node(target)
            G.add_edge(source, target, Ramal=row["Ramal"])
        
        return G
    
    def find_ramal_connection(self, G, ramal_id):
        """
        Find the outfall 'Ramal' for a given ramal_id.
        Args:
            G (DiGraph): The network graph
            ramal_id (str): ID of the ramal to find its outfall
        Returns:
            str: ID of the outfall ramal
        """
        # Encontrar el nodo inicial del ramal
        start_node = next((u for u, v, attrs in G.edges(data=True) if attrs['Ramal'] == ramal_id), None)
        
        if start_node is None:
            return None
        
        current_node = start_node
        while True:
            # Get the successors of the current node
            successors = list(G.successors(current_node))
            current_ramal = current_node.split(".")[0]
            
            # Si no hay sucesores, buscar en predecesores
            if not successors:
                predecessors = list(G.predecessors(current_node))
                if predecessors:
                    return predecessors[0].split(".")[0]
                return current_ramal
            
            # Continuar con el siguiente nodo aguas abajo
            current_node = successors[0]
    
    def get_significant_ramales(self, df, min_number_ramales):
        """
        Determine significant 'Ramales' based on the count of unique connections that meet a minimum threshold.

        Args:
            df (DataFrame): DataFrame with 'Ramal' and 'Conexion' columns.
            min_number_ramales (int): Minimum number of connections to be considered significant.

        Returns:
            connect_dict (dict): Dictionary of significant 'Ramales' and their connections.
        """
        # Gather all unique 'Ramales' from the data
        all_ramales = set(df["Ramal"].unique())
        connect_dict = {}
        
        # Group the data by 'Ramal' for aggregated processing
        grupos = df.groupby("Ramal")
        
        for ramal, grupo in grupos:
            # Filter and split connections to handle multiple entries per 'Ramal'
            conexiones = grupo["Conexion"][grupo["Conexion"] != "None"].str.split(",")
            # Flatten the array of connections
            if conexiones.size > 0:
                all_conexiones = np.concatenate(conexiones.values)
            else:
                break
            
            # Exclude the last connection to avoid self-reference
            last_conexion = conexiones.iloc[-1]
            try:
                filtered_conexiones = all_conexiones[all_conexiones != last_conexion]
            except Exception:
                filtered_conexiones = np.array(list(set(all_conexiones).difference(set(last_conexion))))
            
            # Split and process to get unique parts for significance checking
            try:
                main_parts = pd.Series(filtered_conexiones).str.split(".", expand=True).to_numpy()[:, 0]
            except:
                split_conexiones = np.char.split(filtered_conexiones, ".")
                main_parts = np.array([item[0] for item in split_conexiones])
            unique_conexiones = np.unique(main_parts)
            
            # Add to dictionary if the number of unique connections meets the threshold
            if len(unique_conexiones) >= min_number_ramales:
                connect_dict[ramal] = set(unique_conexiones).intersection(all_ramales)
        
        return connect_dict
    
    def get_map_outRamal(self, G, diccionario_ramales, df_fase):
        """
        Map each 'Ramal' to its corresponding outfall 'Ramal' using network graph analysis.

        Args:
            G (DiGraph): The network graph created from data.
            diccionario_ramales (dict): Dictionary of significant 'Ramales' and their connections.
            df_fase (DataFrame): DataFrame filtered by specific phase or condition.

        Returns:
            outRamal_map (dict): Mapping from each 'Ramal' to its outfall 'Ramal'.
        """
        main_keys = list(diccionario_ramales.keys())
        outRamal_map = {}
        # Process each 'Ramal' group in the DataFrame
        grupos = df_fase.groupby("Ramal")
        for _ramal, grupo in grupos:
            # Determine the starting node and find outfall if not a main key
            if _ramal not in main_keys:
                start_node = grupo["Tramo"].to_numpy()[0].split("-")[0]
                outRamal_map[_ramal] = self.find_outRamal(G, start_node, main_keys)
        # Assign each main key to itself in the map
        for main_key in main_keys:
            outRamal_map[main_key] = main_key
        return outRamal_map
    
    def find_outRamal(self, G, start_node, main_keys):
        """
        Find the outfall 'Ramal' starting from a given node, traversing downstream in the network graph.

        Args:
            G (DiGraph): The network graph.
            start_node (str): Starting node for the traversal.
            main_keys (list): List of main 'Ramales' keys to look for during traversal.

        Returns:
            ramal (str): Identified outfall 'Ramal' or nearest equivalent.
        """
        current_node = start_node
        while True:
            # Get the successors of the current node
            successors = list(G.successors(current_node))
            ramal = current_node.split(".")[0]
            # Return when a main key is reached
            if ramal in main_keys:
                return ramal
            elif not successors:
                # Handle cases with no successors by looking at predecessors
                predecessors = list(G.predecessors(current_node))
                ramal = [_.split(".")[0] for _ in predecessors if _.split(".")[0] in main_keys]
                return ramal[0] if ramal else predecessors[0].split(".")[0]
            
            else:
                # Continue traversal with the next node downstream
                current_node = successors[0]
    
    def make_tramo_index(self, df):
        """
        Update the DataFrame index to a new 'Tramo' based on the last 'Conexion' in each group.

        Args:
            df (DataFrame): DataFrame containing 'Tramo' and 'Conexion' data.

        Returns:
            df (DataFrame): Modified DataFrame with updated 'Tramo' indices.
        """
        # Group the DataFrame by 'Ramal' for processing
        grupos = df.groupby("Ramal")
        for _ramal, grupo in grupos:
            conexion = grupo["Conexion"].iloc[-1]
            indice = grupo["Conexion"].index[-1]
            # Update the 'Tramo' based on the last 'Conexion'
            if conexion not in ["None"]:
                tramo = df.loc[indice, "Tramo"]
                star, end = tramo.split("-")
                new_tramo = "-".join([star, conexion])
                df.loc[indice, "Tramo"] = new_tramo
        # Set the DataFrame index to the updated 'Tramo'
        df.index = df["Tramo"]
        return df
    
    def get_ramal_outfall(self, path, new_column, group_column, min_number_ramales):
        """
        Process a geospatial data file to assign new outfall 'Ramal' based on network analysis.

        Args:
            path (str): Path to the geospatial data file.
            new_column (str): Name for the new column to store outfall 'Ramal'.
            group_column (str): Column name to group the data for analysis.
            min_number_ramales (int): Minimum number of 'Ramales' to consider for analysis.

        Returns:
            df (DataFrame): Processed geospatial DataFrame with new outfall 'Ramal' assigned.
        """
        
        if not isinstance(path, gpd.GeoDataFrame):
            # Read the geospatial data file
            df = gpd.read_file(path, engine="pyogrio")
        else:
            df = path.copy()
            df.reset_index(inplace=True, drop=True)
        # Update the 'Tramo' index for the DataFrame
        df = self.make_tramo_index(df.copy())
        # Initialize the new column with zeros
        df[new_column] = ["0"] * df["Estado"].size
        # Filter DataFrame for new entries
        filtro_nuevo = df["Estado"] == "nuevo"
        df = df[filtro_nuevo]
        for fase in df[group_column].unique():
            # Filter by phase
            filtro_base = df[group_column] == fase
            df_fase = df[filtro_base]
            # Create a network from the phase-specific DataFrame
            G = self.get_network(df_fase)
            # Identify significant 'Ramales'
            diccionario_ramales = self.get_significant_ramales(df_fase.copy(), min_number_ramales)
            # Map each 'Ramal' to its outfall
            outRamal_map = self.get_map_outRamal(G, diccionario_ramales, df_fase)
            # Assign the mapped outfall 'Ramal' to the new column
            df.loc[df_fase.index, new_column] = df_fase["Ramal"].map(outRamal_map)
            df.loc[df_fase.index, 'DependenciaOutFall'] = [self.find_ramal_connection(G, _) for _ in df.loc[df_fase.index, new_column]]  # df_fase.loc[df_fase.index, new_column] = df_fase['Ramal'].map(outRamal_map)  # df_fase.plot(column=new_column)  # plt.show()
        
        return df


# --------------------------------------------------------------------------------------------------------------------------------------------
class TiempoRendimientoEjecuccion:
    def __init__(self):
        self.rendimientos = {'PVC'                                                                  : {0.110: [120, "1 maestro + 3 ayudantes", 6], 0.160: [100, "1 maestro + 3 ayudantes", 6], 0.200: [90, "1 maestro + 3 ayudantes", 6], 0.250: [80, "1 maestro + 4 ayudantes", 6], 0.315: [70, "1 maestro + 4 ayudantes", 6], 0.440: [60, "1 maestro + 4 ayudantes", 6], 0.540: [50, "1 maestro + 5 ayudantes", 6], 0.650: [40, "2 maestros + 6 ayudantes + 1 operador", 6], 0.760: [35, "2 maestros + 6 ayudantes + 1 operador", 6], 0.875: [30, "2 maestros + 6 ayudantes + 1 operador", 6],
                                                                                                       0.975: [25, "2 maestros + 7 ayudantes + 1 operador", 6]},
                             'PEAD': {0.200: [48, "1 maestro + 4 ayudantes", 6], 0.315: [48, "1 maestro + 4 ayudantes", 6], 0.400: [48, "1 maestro + 4 ayudantes", 6], 0.500: [36, "1 maestro + 5 ayudantes", 6], 0.630: [36, "2 maestros + 6 ayudantes + 1 operador", 6], 0.710: [36, "2 maestros + 6 ayudantes + 1 operador", 6], 0.800: [24, "2 maestros + 7 ayudantes + 1 operador", 6], 0.900: [24, "2 maestros + 7 ayudantes + 1 operador", 6], 1.000: [12, "2 maestros + 8 ayudantes + 1 operador", 6], 1.200: [12, "2 maestros + 8 ayudantes + 1 operador", 6]},
                             'PRFV': {0.300: [70, "1 maestro + 4 ayudantes", 6], 0.400: [60, "1 maestro + 4 ayudantes", 6], 0.500: [50, "1 maestro + 5 ayudantes", 6], 0.600: [40, "2 maestros + 6 ayudantes + 1 operador", 6], 0.700: [35, "2 maestros + 6 ayudantes + 1 operador", 6], 0.800: [30, "2 maestros + 7 ayudantes + 1 operador", 6], 0.900: [25, "2 maestros + 7 ayudantes + 1 operador", 6], 1.000: [20, "2 maestros + 8 ayudantes + 1 operador", 6], 1.200: [15, "2 maestros + 8 ayudantes + 1 operador", 6], 1.500: [12, "2 maestros + 8 ayudantes + 1 operador", 6],
                                      2.000: [10, "2 maestros + 8 ayudantes + 1 operador", 6]}, 'HS': {0.160: [50, "1 maestro + 3 ayudantes + 1 operador", 1], 0.200: [45, "1 maestro + 3 ayudantes + 1 operador", 1], 0.300: [40, "1 maestro + 4 ayudantes + 1 operador", 1], 0.400: [35, "1 maestro + 4 ayudantes + 1 operador", 1], 0.500: [30, "1 maestro + 5 ayudantes + 1 operador", 1], 0.600: [25, "2 maestros + 6 ayudantes + 1 operador", 1]}}
    
    def get_rendimiento_hormigon_in_situ(self, diametro, seccion_dim):
        
        base_seccion, altura_seccion = seccion_dim.split('x')
        base_seccion, altura_seccion = float(base_seccion), float(altura_seccion)
        
        """Calcula equipo y rendimiento para secciones de hormigón in situ"""
        if diametro <= 0.5:
            espesor = 0.15
            rendimiento = 12  # m3/dia
            return {'rendimiento': rendimiento / ((base_seccion * espesor * 2) + (altura_seccion + espesor * 2) * espesor), 'equipo': "1 maestro + 4 ayudantes + 2 carpinteros + 1 operador mixer", 'actividades': {'encofrado': '2 carpinteros - 1 día cada 8m', 'armado': '2 fierreros - 1 día cada 8m', 'fundido': '1 maestro + 4 ayudantes - 8m por día', 'fraguado': '7 días obligatorios'}}
        elif diametro <= 1.0:
            espesor = 0.25
            rendimiento = 10  # m3/dia
            return {'rendimiento': rendimiento / ((base_seccion * espesor * 2) + (altura_seccion + espesor * 2) * espesor), 'equipo': "2 maestros + 6 ayudantes + 3 fierreros + 3 carpinteros + 1 operador mixer + 1 operador grúa", 'actividades': {'encofrado': '3 carpinteros - 1 día cada 6m', 'armado': '3 fierreros - 1 día cada 6m', 'fundido': '2 maestros + 6 ayudantes - 6m por día', 'fraguado': '7 días obligatorios'}}
        else:
            espesor = 0.35
            rendimiento = 8  # m3/dia
            return {'rendimiento': rendimiento / ((base_seccion * espesor * 2) + (altura_seccion + espesor * 2) * espesor), 'equipo': "2 maestros + 8 ayudantes + 4 fierreros + 4 carpinteros + 1 operador mixer + 1 operador grúa", 'actividades': {'encofrado': '4 carpinteros - 1 día cada 4m', 'armado': '4 fierreros - 1 día cada 4m', 'fundido': '2 maestros + 8 ayudantes - 4m por día', 'fraguado': '7 días obligatorios'}}
    
    def tiempo_ejeuccion(self, df_tramos):
        """
        Calcula los días de construcción para diferentes tipos de tubería
        
        Args:
            df_tramos: DataFrame con columnas ['longitud', 'material', 'diametro'] (en metros)
        Returns:
            DataFrame con resultados del cálculo
        """
        dias = []
        equipo = []
        detalles = []
        for _, row in df_tramos.iterrows():
            material = row['material']
            diametro = row['diametro']
            longitud = row['longitud']
            seccion = row['seccion']
            seccion_dim = row['seccion_dim']
            
            if material == 'HA' and seccion in ['rectangular']:
                
                info = self.get_rendimiento_hormigon_in_situ(diametro, seccion_dim)
                dias_totales = longitud / info['rendimiento']
                
                dias.append(round(dias_totales, 2))
                equipo.append(info['equipo'])
                detalles.append(info['actividades'])
            
            elif material in self.rendimientos:
                diametros = list(self.rendimientos[material].keys())
                closest_dia = min(diametros, key=lambda x: abs(x - diametro))
                rendimiento_diario = self.rendimientos[material][closest_dia][0]
                equipo_necesario = self.rendimientos[material][closest_dia][1]
                longitud_tubo = self.rendimientos[material][closest_dia][2]
                
                n_uniones = (longitud / longitud_tubo) - 1
                tiempo_por_union = 0.125
                tiempo_total = (longitud / rendimiento_diario) + (n_uniones * tiempo_por_union)
                
                dias.append(round(tiempo_total, 2))
                equipo.append(equipo_necesario)
                detalles.append(f"Tubos de {longitud_tubo}m - {int(n_uniones)} uniones")
            
            else:
                dias.append(None)
                equipo.append(None)
                detalles.append(None)
        
        return pd.DataFrame({'Longitud': np.concatenate(df_tramos['longitud'].to_numpy()), 'Material': np.concatenate(df_tramos['material'].to_numpy()), 'Diámetro': np.concatenate(df_tramos['diametro'].to_numpy()), 'Días': dias, 'Equipo_Necesario': equipo, 'Detalles': detalles})


class FrentesTrabajoAlcantarillado:
    def __init__(self, df):
        self.df = df
        self.resumen = None
        self.cronograma = None
        self.max_frentes_simultaneos = 3
    
    def procesar_frente(self, front_df):
        resultados = []
        rendimineto = TiempoRendimientoEjecuccion()
        for diametro in front_df["D_ext"].unique():
            df_diametro = front_df[front_df["D_ext"] == diametro]
            
            material_arr = df_diametro['Material'].to_numpy()
            diametro_arr = seccion_str2float(df_diametro['D_ext']).astype(float)
            seccion_dim_arr = df_diametro['D_ext'].to_numpy()
            longitud_arr = df_diametro['L'].to_numpy()
            seccion_arr = df_diametro['Seccion'].to_numpy()
            s = pd.DataFrame([material_arr, diametro_arr, seccion_arr, seccion_dim_arr, longitud_arr]).T
            s.columns = [['material', 'diametro', 'seccion', 'seccion_dim', 'longitud']]
            resultados_rendimiento = rendimineto.tiempo_ejeuccion(s)
            
            resultados.append({"Fase": front_df["Fase"].iloc[0], "Frente de Trabajo": front_df["RamalOutFall"].iloc[0], "Dependencia": front_df["DependenciaOutFall"].iloc[0], "Diámetro": diametro, "Longitud Total (m)": round(df_diametro["L"].sum(), 2), "Tiempo Estimado (días)": resultados_rendimiento['Días'].sum(), })
        return resultados
    
    def generar_resumen(self):
        resultados = []
        for _, phase_df in self.df.groupby("Fase"):
            for _, front_df in phase_df.groupby("RamalOutFall"):
                resultados.extend(self.procesar_frente(front_df))
        
        self.resumen = pd.DataFrame(resultados)
        self.resumen = self.resumen[["Fase", "Frente de Trabajo", "Dependencia", "Diámetro", "Longitud Total (m)", "Tiempo Estimado (días)"]]
        return self.resumen
    
    def generar_cronograma(self, output_dir):
        if self.resumen is None:
            self.generar_resumen()
        
        # Set start date
        gantt_data = []
        end_date_max = []
        current_date_fase = datetime(2026, 1, 9)
        # current_date_fase = datetime(2025, 6, 1)
        
        for fase, grupo in self.resumen.groupby('Fase'):
            grupo = grupo.reset_index(drop=True)  # Reset index at fase level
            current_date_fase = current_date_fase if len(end_date_max) == 0 else max(end_date_max)
            already_done_frente = set()
            pending_dependent_fronts = {}  # Store fronts waiting to start in parallel
            active_fronts = set()  # Keep track of currently active fronts
            cond = True
            end_frente_date = {}
            
            while cond:
                grupos = {name: group.reset_index(drop=True) for name, group in grupo.groupby('Frente de Trabajo')}
                
                # First process independent fronts (self-dependent)
                for frente, grupo_frente in grupos.items():
                    # Skip if this front is already processed
                    if frente in already_done_frente:
                        continue
                    
                    # Process diameter ordering
                    split_diameter = grupo_frente['Diámetro'].str.split('x', expand=True)
                    if isinstance(split_diameter, pd.DataFrame):
                        
                        if split_diameter.shape[1] == 1:
                            grupo_frente['order'] = np.where(split_diameter[0].notna(), split_diameter[0], grupo_frente['Diámetro'])
                        else:
                            grupo_frente['order'] = np.where(split_diameter[1].notna(), split_diameter[1], grupo_frente['Diámetro'])
                    
                    else:
                        grupo_frente['order'] = split_diameter
                    
                    order_index = natsort.index_natsorted(grupo_frente['order'], reverse=True)
                    grupo_frente = grupo_frente.loc[order_index].reset_index(drop=True)
                    
                    # If front depends on itself, process it first
                    if frente == grupo_frente.loc[0, 'Dependencia']:
                        current_date = current_date_fase
                        
                        for idx, row in grupo_frente.iterrows():
                            task_name = f'Frente:{row["Frente de Trabajo"]}-Ø:{row["Diámetro"]}-L:{round(row["Longitud Total (m)"], 1)}m'
                            duration = row['Tiempo Estimado (días)']
                            
                            end_date = current_date + timedelta(days=max(duration, 1))
                            gantt_data.append({'Task': task_name, 'Start': current_date.strftime('%Y-%m-%d'), 'Finish': end_date.strftime('%Y-%m-%d'), 'Resource': f'Frente:{row["Frente de Trabajo"]}-Ø:{row["Diámetro"]}', 'Fase': fase})
                            
                            current_date = end_date
                            end_frente_date[frente] = end_date
                            end_date_max.append(end_date)
                        
                        already_done_frente.add(frente)
                        # Add dependent fronts to pending
                        for dep_frente, dep_grupo in grupos.items():
                            if dep_grupo.loc[0, 'Dependencia'] == frente and dep_frente not in already_done_frente:
                                if frente not in pending_dependent_fronts:
                                    pending_dependent_fronts[frente] = []
                                pending_dependent_fronts[frente].append(dep_frente)
                
                # Process pending dependent fronts (up to 3 in parallel)
                for dependency, waiting_fronts in list(pending_dependent_fronts.items()):
                    if dependency in end_frente_date and len(active_fronts) < 3:
                        start_date = end_frente_date[dependency]
                        
                        while waiting_fronts and len(active_fronts) < 3:
                            next_front = waiting_fronts.pop(0)
                            if next_front not in already_done_frente:
                                active_fronts.add(next_front)
                                grupo_frente = grupos[next_front]
                                current_date = start_date
                                
                                for idx, row in grupo_frente.iterrows():
                                    task_name = f'Frente:{row["Frente de Trabajo"]}-Ø:{row["Diámetro"]}-L:{round(row["Longitud Total (m)"], 1)}m'
                                    duration = row['Tiempo Estimado (días)']
                                    
                                    end_date = current_date + timedelta(days=max(duration, 1))
                                    gantt_data.append({'Task': task_name, 'Start': current_date.strftime('%Y-%m-%d'), 'Finish': end_date.strftime('%Y-%m-%d'), 'Resource': f'Frente:{row["Frente de Trabajo"]}-Ø:{row["Diámetro"]}', 'Fase': fase})
                                    
                                    current_date = end_date
                                    end_frente_date[next_front] = end_date
                                    end_date_max.append(end_date)
                                
                                already_done_frente.add(next_front)
                        
                        if not waiting_fronts:
                            del pending_dependent_fronts[dependency]
                
                # Update active_fronts based on completion dates
                current_date = max(end_date_max) if end_date_max else current_date_fase
                active_fronts = {front for front in active_fronts if end_frente_date.get(front) and end_frente_date[front] > current_date}
                
                # Check if all fronts are processed
                if already_done_frente == set(grupos.keys()):
                    cond = False
        
        # Create the timeline
        df = pd.DataFrame(gantt_data)
        df['Start'] = pd.to_datetime(df['Start'])
        df['Finish'] = pd.to_datetime(df['Finish'])
        # Calculate duration
        df['Duration'] = (df['Finish'] - df['Start']).dt.days
        
        self.cronograma = df.copy()
        
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Fase", color_discrete_sequence=px.colors.qualitative.Plotly,  # More vibrant color palette
                          hover_data={'Start'   : '|%Y-%m-%d',  # Format start date
                                      'Finish'  : '|%Y-%m-%d',  # Format end date
                                      'Duration': ':.0f dias',  # Show duration in days
                                      # 'Resource': True,  # Include resource information
                                      'Fase'    : True  # Include phase information
                                      }, title="Cronograma de Ejecuccion", labels={"Task": "Frentes de Trabajo", "Fase": "Componente"})
        
        fig.update_yaxes(autorange="reversed")
        
        # # # Guardar cronograma visual
        
        output_path = os.path.join(output_dir, "cronograma.html")
        fig.write_html(output_path)
        
        # fig = self.visualizar_cronograma(gantt_data)  # output_path = os.path.join(output_dir, "cronograma_nuevo.html")  # fig.write_html(output_path)
    
    def guardar_resultados(self, output_dir="resultados"):
        """
        Guarda todos los resultados en una ubicación específica
        
        Args:
            output_dir (str): Directorio donde se guardarán los resultados
        """
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Generar resumen y cronograma si no existen
        if self.resumen is None:
            self.generar_resumen()
        if self.cronograma is None:
            self.generar_cronograma(output_dir)
        
        # Guardar resumen y cronograma en un solo Excel
        with pd.ExcelWriter(os.path.join(output_dir, "resultados_frentes_trabajo.xlsx")) as writer:
            self.resumen.to_excel(writer, sheet_name='Resumen', index=False)
            self.cronograma.to_excel(writer, sheet_name='Cronograma', index=False)
    
    def visualizar_cronograma(self, gantt_data):
        df = pd.DataFrame(gantt_data)
        
        # Convert dates to datetime
        df['Start'] = pd.to_datetime(df['Start'])
        df['Finish'] = pd.to_datetime(df['Finish'])
        
        # Calculate duration for each task
        df['Duration'] = (df['Finish'] - df['Start']).dt.days
        
        # Calculate critical path
        df['Is_Critical'] = df.groupby('Task')['Duration'].transform(lambda x: x >= x.quantile(0.75))
        
        # Create color map for different task types
        task_colors = {task: 'rgb(46, 137, 205)' if i % 2 == 0 else 'rgb(114, 44, 121)' for i, task in enumerate(df['Task'].unique())}
        
        # Create the figure
        fig = go.Figure()
        
        # Sort tasks by start date
        df = df.sort_values('Start')
        
        # Create positions for each unique task
        tasks = df['Task'].unique()
        task_positions = {task: i for i, task in enumerate(tasks)}
        
        # Add tasks as horizontal bars
        for task in df['Task'].unique():
            task_data = df[df['Task'] == task]
            
            # Add main task bars
            fig.add_trace(go.Bar(y=[task_positions[task]] * len(task_data),  # Position on y-axis
                                 x=task_data['Duration'],  # Duration in days
                                 base=task_data['Start'],  # Start date
                                 name=task, orientation='h',  # Make bars horizontal
                                 marker_color=task_colors[task], marker_line_color='rgba(0, 0, 0, 0.5)', marker_line_width=1, customdata=np.stack((task_data['Task'], task_data['Resource'], task_data['Duration'], task_data['Start'].dt.strftime('%Y-%m-%d'), task_data['Finish'].dt.strftime('%Y-%m-%d')), axis=-1), hovertemplate='<br>'.join(['<b>%{customdata[0]}</b>', 'Recurso: %{customdata[1]}', 'Duración: %{customdata[2]} días', 'Inicio: %{customdata[3]}', 'Fin: %{customdata[4]}', '<extra></extra>'])))
            
            # Add markers for critical path tasks
            critical_tasks = task_data[task_data['Is_Critical']]
            if not critical_tasks.empty:
                fig.add_trace(go.Scatter(x=critical_tasks['Start'], y=[task_positions[task]] * len(critical_tasks), mode='markers', name='Ruta Crítica', marker=dict(symbol='star', size=8, color='red', line=dict(width=1, color='red')), showlegend=True))
        
        # Calculate date range for x-axis
        date_range = pd.date_range(start=df['Start'].min(), end=df['Finish'].max(), freq='MS')
        
        # Update layout
        fig.update_layout(title={'text': 'Cronograma de Ejecución', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=24)}, barmode='relative', height=50 * len(tasks) + 200,  # Dynamic height based on number of tasks
                          showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(title='Periodo de Ejecución', gridcolor='rgba(0, 0, 0, 0.1)', griddash='dash', ticktext=[d.strftime('%b %Y') for d in date_range], tickvals=date_range, tickangle=45, type='date'), yaxis=dict(title='Actividades', gridcolor='rgba(0, 0, 0, 0.1)', griddash='dash', ticktext=list(tasks), tickvals=list(task_positions.values()), autorange='reversed'  # Reverse y-axis to show tasks from top to bottom
                                                                                                                                                                                                                                                                                                                                   ), plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=200, r=50, t=100, b=100)  # Increased left margin for task names
                          )
        
        # Add shapes for month divisions and current date
        shapes = []
        
        # Add vertical lines for months
        for date in date_range:
            shapes.append(dict(type='line', x0=date, x1=date, y0=min(task_positions.values()) - 0.5, y1=max(task_positions.values()) + 0.5, line=dict(color='rgba(0, 0, 0, 0.2)', dash='dash')))
        
        # Add vertical line for current date
        current_date = pd.Timestamp.now()
        if df['Start'].min() <= current_date <= df['Finish'].max():
            shapes.append(dict(type='line', x0=current_date, x1=current_date, y0=min(task_positions.values()) - 0.5, y1=max(task_positions.values()) + 0.5, line=dict(color='red', width=2), name='Fecha Actual'))
            
            # Add annotation for current date
            fig.add_annotation(x=current_date, y=max(task_positions.values()) + 0.5, text='Fecha Actual', showarrow=False, yshift=10)
        
        fig.update_layout(shapes=shapes)
        
        return fig







class SewerConstructionCost:
    """
    Calcula el costo de construcción de un sistema de alcantarillado.
    
    Args:
        vector_path: Ruta al archivo GPKG con los ramales diseñados
        tipo: Tipo de alcantarillado ('SANITARIO', 'PLUVIAL', 'COMBINADO')
        fase: Fase a filtrar (default: 'GENERAL')
        domiciliarias_vector_path: Ruta al shapefile de domiciliarias (opcional)
        **kwargs: Parámetros opcionales para sobreescribir valores por defecto
    
    Uso:
        calc = SewerConstructionCost("ramales.gpkg", "SANITARIO", fase="F1")
        total = calc.run()
    """
    
    def __init__(self, vector_path: Path, tipo: str, fase: str = 'GENERAL',
                 domiciliarias_vector_path: str = None, base_precios: str = None,  **kwargs):
        self.vector_path = vector_path
        self.tipo = tipo
        self.fase = fase
        self.domiciliarias_vector_path = domiciliarias_vector_path
        self.base_precios = base_precios
        
        # Generar tabla de anchos de zanja
        self.trench_widths = TrenchWidthCalculator()
        df_trench_widths = self.trench_widths.generate_diameter_table(vg.diametro_interno_externo_pypiper)
        self.tabla_ancho = self.trench_widths.round_and_convert(df_trench_widths)
        
        # Construir parameters_dict internamente con valores por defecto
        self.parameters_dict = {
            'vector_path': vector_path,
            'base_precios':self.base_precios,
            'tabla_ancho': self.tabla_ancho,
            'tabla_taludes': vg.tabla_taludes,
            'tabla_tipo_suelo': vg.tabla_tipo_suelo,
            'altura_minima_entibado_discontinuo_madera': kwargs.get('altura_min_entibado_disc', 1.5),
            'altura_minima_entibado_continuo_madera': kwargs.get('altura_min_entibado_cont', 2.5),
            'altura_minima_entibado_continuo_metalico': kwargs.get('altura_min_entibado_metal', 3.0),
            'distancia_desalojo': kwargs.get('distancia_desalojo', 25),
            'porcentaje_esponjamiento': kwargs.get('porcentaje_esponjamiento', 0.3),
            'porcentage_desbroce': kwargs.get('porcentage_desbroce', 0.2),
            'distancia_acarreo_manual': kwargs.get('distancia_acarreo_manual', 5),
            'aumento_de_cantidades': kwargs.get('aumento_de_cantidades', 0.15),
            'filtro': {'column': 'Fase', 'value': fase},
            'domiciliarias': {
                'numero_domiciliarias': kwargs.get('numero_domiciliarias', None),
                'porcentaje_acarreo_manual': 0.8,
                'porcentaje_acarreo_mecanica': 0.2,
                'profundidad_zanja': 2,
                'ancho_zanja': 0.7,
                'longitud_tuberia': 6,
                'porcentaje_relleno_compactado': 0.2,
                'porcentaje_material_mejoramiento': 0.8,
                'domiciliarias_vector_path': domiciliarias_vector_path,
            },
            'sumideros': {
                'distancia_entre_sumideros': None,
                'porcentaje_acarreo_manual': 0.2,
                'porcentaje_acarreo_mecanica': 0.8,
                'profundidad_zanja': 2,
                'ancho_zanja': 0.7,
                'longitud_tuberia': 6,
                'porcentaje_relleno_compactado': 0.2,
                'porcentaje_material_mejoramiento': 0.8,
            },
            'interferencias_agua_potable': {
                'longitud_tuberia': 6,
                'longitud_tuberia_domiciliaria': 4,
                'porcentaje_reparacion_tuberia': None,
                'porcentaje_reparacion_domiciliaria': 0.25,
            },
            'tuberia_desvio_temporal': {
                'porcentaje_longitud_desvio': 0.2,
                'numero_de_usos': 6,
            },


        }
        self.data = {
            "REPLANTEO Y NIVELACION": {
                "01.001.4.04": ("REPLANTEO Y NIVELACION DE EJES (KM)", "Km"),
            },

            "EXCAVACION A MANO": {
                "01.003.4.01": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (EN TIERRA)", "m3"),
                "01.003.4.02": ("EXCAVACION ZANJA A MANO H=2,76-3,99M (EN TIERRA)", "m3"),
                "01.003.4.03": ("EXCAVACION ZANJA A MANO H=4,00-6,00M (EN TIERRA)", "m3"),
                "01.003.4.05": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (CONGLOMERADO)", "m3"),
                "01.003.4.06": ("EXCAVACION ZANJA A MANO H=2,76-3,99M (CONGLOMERADO)", "m3"),
                "01.003.4.07": ("EXCAVACION ZANJA A MANO H=4,00-6,00M (CONGLOMERADO)", "m3"),
                "01.003.4.13": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (ROCA) INCL. EQUIPO LIVIANO", "m3"),
                "01.003.4.14": ("EXCAVACION ZANJA A MANO H=2,76-3,99M (ROCA) INCL. EQUIPO LIVIANO", "m3"),
                "01.003.4.15": ("EXCAVACION ZANJA A MANO H=4,00-6,00M (ROCA) INCL.  EQUIPO LIVIANO", "m3"),
            },

            "EXCAVACION MECANICA": {
                "01.003.4.24": ("EXCAVACION ZANJA A MAQUINA H=0,00-2,75M (EN TIERRA)", "m3"),
                "01.003.4.25": ("EXCAVACION ZANJA A MAQUINA H=2,76-3,99M (EN TIERRA)", "m3"),
                "01.003.4.26": ("EXCAVACION ZANJA A MAQUINA H=4,00-6,00M (EN TIERRA)", "m3"),
                "01.003.4.27": ("EXCAVACION ZANJA A MAQUINA H>6,00M (EN TIERRA)", "m3"),
                "01.003.4.28": ("EXCAVACION ZANJA A MAQUINA H=0,00-2,75M (CONGLOMERADO)", "m3"),
                "01.003.4.29": ("EXCAVACION ZANJA A MAQUINA H=2,76-3,99M (CONGLOMERADO)", "m3"),
                "01.003.4.31": ("EXCAVACION ZANJA A MAQUINA H=4,00-6,00M (CONGLOMERADO)", "m3"),
                "01.003.4.3": ("EXCAVACION ZANJA A MAQUINA H>6,00M (CONGLOMERADO)", "m3"),
                "01.003.4.42": ("EXCAVACION ZANJA A MAQUINA H=0,00-2,75M (ROCA)", "m3"),
                "01.003.4.43": ("EXCAVACION ZANJA A MAQUINA H=2,76-3,99M (ROCA)", "m3"),
                "01.003.4.44": ("EXCAVACION ZANJA A MAQUINA H=4,00-6,00M (ROCA)", "m3"),
                "01.003.4.45": ("EXCAVACION ZANJA A MAQUINA H>6,00M (ROCA)", "m3"),
            },

            "BOMBEO DE AGUA": {
                "500306": ("BOMBEO DE AGUA (SE PAGARA POR HORA)", "u"),
                "500481": ("BOMBEO DE ACHIQUE (SE PAGARA POR HORA)", "u"),
            },

            "DESBROCE Y LIMPIEZA": {
                "01.002.4.01": ("DESBROCE Y LIMPIEZA", "m2"),
            },

            "ENTIBADO": {
                "01.008.4.17": ("ENTIBADO DISCONTINUO (APUNTALAMIENTO) ZANJA - RETORNABLE", "m2"),
                "500485": ("ENTIBADO CONTINUO DE MADERA PARA ZANJA  - RETORNABLE (SEGUN ESPECIFICACION TECNICA)", "m2"),
                "500408": ("ENTIBADO CONTINUO DE ZANJA (METALICO)  - RETORNABLE", "m2"),
            },

            "RELLENO Y DESALOJO": {
                "01.005.4.01": ("RELLENO COMPACTADO MATERIAL DE EXCAVACION  - EQUIPO LIVIANO", "m3"),
                "01.016.4.09": ("MATERIAL DE MEJORAMIENTO", "m3"),
                "01.016.4.99": ("BASE CLASE 2 -EN ZANJAS DE REDES (EQUIPO LIVIANO) (MATERIAL/TRANSPORTE/TENDIDO/COMPACTADO)", "m3"),
                "01.016.4101": ("SUB-BASE CLASE 2 -EN ZANJAS DE REDES (EQUIPO LIVIANO) (MATERIAL/TRANSPORTE/TENDIDO/COMPACTADO)", "m3"),
                "01.016.4.56": ("CAMA DE ARENA", "m3"),
                "01.007.4.80": ("TRANSPORTE MANUAL HORIZONTAL (CARRETILLA O SIMILAR) (SE PAGARÁ POR M3XM)", "u"),
                "01.007.4.02": ("ACARREO MECANICO HASTA 1 KM (CARGA,TRANSPORTE,VOLTEO)", "m3"),
                "01.007.4.63": ("SOBREACARREO MATERIAL NO UTILIZABLE A BOTADERO (TRANSPORTE/MEDIOS MECÁNICOS) (SE PAGARA EN M3-KM ) - NO INCL.CARGA", "u"),
            },

            "MEJORAMIENTO DE FONDO DE ZANJA": {
                "01.003.4.01": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (EN TIERRA)", "m3"),
                "01.003.4.02": ("EXCAVACION ZANJA A MANO H=2,76-3,99M (EN TIERRA)", "m3"),
                "01.003.4.03": ("EXCAVACION ZANJA A MANO H=4,00-6,00M (EN TIERRA)", "m3"),

                "01.003.4.05": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (CONGLOMERADO)", "m3"),
                "01.003.4.06": ("EXCAVACION ZANJA A MANO H=2,76-3,99M (CONGLOMERADO)", "m3"),
                "01.003.4.07": ("EXCAVACION ZANJA A MANO H=4,00-6,00M (CONGLOMERADO)", "m3"),

                "01.003.4.13": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (ROCA) INCL. EQUIPO LIVIANO", "m3"),
                "01.003.4.14": ("EXCAVACION ZANJA A MANO H=2,76-3,99M (ROCA) INCL. EQUIPO LIVIANO", "m3"),
                "01.003.4.15": ("EXCAVACION ZANJA A MANO H=4,00-6,00M (ROCA) INCL.  EQUIPO LIVIANO", "m3"),

                "01.003.4.24": ("EXCAVACION ZANJA A MAQUINA H=0,00-2,75M (EN TIERRA)", "m3"),
                "01.003.4.25": ("EXCAVACION ZANJA A MAQUINA H=2,76-3,99M (EN TIERRA)", "m3"),
                "01.003.4.26": ("EXCAVACION ZANJA A MAQUINA H=4,00-6,00M (EN TIERRA)", "m3"),
                "01.003.4.27": ("EXCAVACION ZANJA A MAQUINA H>6,00M (EN TIERRA)", "m3"),

                "01.003.4.28": ("EXCAVACION ZANJA A MAQUINA H=0,00-2,75M (CONGLOMERADO)", "m3"),
                "01.003.4.29": ("EXCAVACION ZANJA A MAQUINA H=2,76-3,99M (CONGLOMERADO)", "m3"),
                "01.003.4.31": ("EXCAVACION ZANJA A MAQUINA H=4,00-6,00M (CONGLOMERADO)", "m3"),
                "01.003.4.3": ("EXCAVACION ZANJA A MAQUINA H>6,00M (CONGLOMERADO)", "m3"),

                "01.003.4.42": ("EXCAVACION ZANJA A MAQUINA H=0,00-2,75M (ROCA)", "m3"),
                "01.003.4.43": ("EXCAVACION ZANJA A MAQUINA H=2,76-3,99M (ROCA)", "m3"),
                "01.003.4.44": ("EXCAVACION ZANJA A MAQUINA H=4,00-6,00M (ROCA)", "m3"),
                "01.003.4.45": ("EXCAVACION ZANJA A MAQUINA H>6,00M (ROCA)", "m3"),

                "01.005.4.06": ("RELLENO CON GRAVA", "m3"),
                "01.005.4.11": ("RELLENO CON PIEDRA", "m3"),
                "500492": ("GEOMALLA BIAXIAL RESISTENCIA MINIMA DE 50KN (PROVISION E INSTALACION, INC. ANCLAJE)", "m2"),
                "01.011.4.32": ("INYECCION DE LECHADA DE CEMENTO-INCL.PERFORACION (TONELADA DE CEMENTO)", "Tn"),
            },

            "SUMINISTRO E INSTALACION TUBERIA PEAD": {
                "500308": ("SUM, INST TUBERIA DE PEAD PE 100 D= 200 MM PN6 SDR 26", "m"),
                "500309": ("SUM, INST TUBERIA DE PEAD PE 100 D= 315 MM PN6 SDR 26", "m"),
                "500311": ("SUM, INST TUBERIA DE PEAD PE 100 D= 400 MM PN6 SDR 26", "m"),
                "500313": ("SUM, INST TUBERIA DE PEAD PE 100 D= 500 MM PN6 SDR 26", "m"),
                "500315": ("SUM, INST TUBERIA DE PEAD PE 100 D= 630 MM PN6 SDR 26", "m"),
                "500317": ("SUM, INST TUBERIA DE PEAD PE 100 D= 710 MM PN6 SDR 26", "m"),
                "500319": ("SUM, INST TUBERIA DE PEAD PE 100 D= 800 MM PN6 SDR 26", "m"),
                "500321": ("SUM, INST TUBERIA DE PEAD PE 100 D= 900 MM PN6 SDR 26", "m"),
                "500323": ("SUM, INST TUBERIA DE PEAD PE 100 D= 1000 MM PN6 SDR 26", "m"),
                "500325": ("SUM, INST TUBERIA DE PEAD PE 100 D= 1200 MM PN6 SDR 26", "m"),
            },

            "SUMINISTRO E INSTALACION TUBERIA PVC": {
                "500264": ("TUBERIA PVC UE ALCANTARILLADO D.I.N. 200mm (MAT.TRAN.INST)", "m"),
                "500266": ("TUBERIA PVC UE ALCANTARILLADO D.I.N. 300mm (MAT.TRAN.INST)", "m"),
                "500267": ("TUBERIA PVC UE ALCANTARILLADO D.I.N. 400mm (MAT.TRAN.INST)", "m"),
                "500273": ("TUBERIA PVC UE ALCANTARILLADO D.I.N. 500mm (MAT.TRAN.INST)", "m"),
                "500269": ("TUBERIA PVC UE ALCANTARILLADO D.I.N. 600mm (MAT.TRAN.INST)", "m"),
                "500270": ("TUBERIA PVC UE ALCANTARILLADO D.I.N. 700mm (MAT.TRAN.INST)", "m"),
                "500271": ("TUBERIA PVC UE ALCANTARILLADO D.I.N. 800mm (MAT.TRAN.INST)", "m"),
                "500274": ("TUBERIA PVC UE ALCANTARILLADO D.I.N. 900mm (MAT.TRAN.INST)", "m"),
            },

            "SUMINISTRO E INSTALACION TUBERIA HA": {
                "512405": ("TUBERIA HORMIGON ARMADO ALCANTARILLADO D.I.N. 500mm (MAT.TRAN.INST)", "m"),
                "512406": ("TUBERIA HORMIGON ARMADO ALCANTARILLADO D.I.N. 600mm (MAT.TRAN.INST)", "m"),
                "512407": ("TUBERIA HORMIGON ARMADO ALCANTARILLADO D.I.N. 700mm (MAT.TRAN.INST)", "m"),
                "512408": ("TUBERIA HORMIGON ARMADO ALCANTARILLADO D.I.N. 800mm (MAT.TRAN.INST)", "m"),
                "512409": ("TUBERIA HORMIGON ARMADO ALCANTARILLADO D.I.N. 900mm (MAT.TRAN.INST)", "m"),
                "512410": ("TUBERIA HORMIGON ARMADO ALCANTARILLADO D.I.N. 1000mm (MAT.TRAN.INST)", "m"),
                "512411": ("TUBERIA HORMIGON ARMADO ALCANTARILLADO D.I.N. 1100mm (MAT.TRAN.INST)", "m"),
                "512412": ("TUBERIA HORMIGON ARMADO ALCANTARILLADO D.I.N. 1200mm (MAT.TRAN.INST)", "m"),
                "512415": ("TUBERIA HORMIGON ARMADO ALCANTARILLADO D.I.N. 1500mm (MAT.TRAN.INST)", "m"),
                "512418": ("TUBERIA HORMIGON ARMADO ALCANTARILLADO D.I.N. 1800mm (MAT.TRAN.INST)", "m"),
            },

            "COLECTOR DE HORMIGON ARMADO": {
                "500327": ("REPLANTILLO DE PIEDRA, ESPESOR=20 CM", "m2"),
                "01.011.4.30": ("HORMIGON PREMEZCLADO REPLANTILLO FC=180 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.010.4.13": ("ENCOFRADO/DESENCOFRADO METALICO RECTO", "m2"),
                "01.011.4118": ("HORMIGON PREMEZCLADO FC=280 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.009.4.01": ("ACERO REFUERZO FY=4200 KG/CM2 (SUMINISTRO, CORTE Y COLOCADO)", "kg"),
            },

            "CANALES ABIERTOS Y REJILLAS": {
                "500327": ("REPLANTILLO DE PIEDRA, ESPESOR=20 CM", "m2"),
                "01.011.4.30": ("HORMIGON PREMEZCLADO REPLANTILLO FC=180 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.010.4.13": ("ENCOFRADO/DESENCOFRADO METALICO RECTO", "m2"),
                "01.011.4118": ("HORMIGON PREMEZCLADO FC=280 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.009.4.83": ("MALLA ELECTROSOLDADA FY=5000KG/CM2 (PROVISION Y MONTAJE)", "kg"),
                "500332": ("SUM. INST. REJILLA PLASTICA PARA CANALES 1000MM X 360MM, (INCLUYE, SISTEMA DE FIJACION TIPO ARGOLLA  Y SEGURO ANTIROBO)-RESISTENCIA 125KN", "u"),
            },

            "PERFORACION HORIZONTAL DIRIGIDA": {
                "16.002.4.13": ("TUBERÍA PEAD PE 100, D=400MM PN10 SDR 17 – AASS, AALL / SUELOS TIPO 1 (MATERIAL, TRANSPORTE, INSTALACIÓN", "m"),
                "16.002.4.15": ("TUBERÍA PEAD PE 100, D=500MM PN10 SDR 17 – AASS, AALL / SUELOS TIPO 1 (MATERIAL, TRANSPORTE, INSTALACIÓN", "m"),
                "16.002.4.18": ("TUBERÍA PEAD PE 100, D=710MM PN10 SDR 17 – AASS, AALL / SUELOS TIPO 1 (MATERIAL, TRANSPORTE, INSTALACIÓN", "m"),
                "11.012.4.05": ("MAPEAMIENTO DE INTERFERENCIAS GEORADAR PARA TECNOLOGIA SIN ZANJA (M)", "m"),
                "16.003.4.05": ("DESALOJO DE LODOS DE PERFORACION (M3-KM)", "u"),
            },

            "TRINCHERAS PERFORACION HORIZONTAL DIRIGIDA": {
                "05.012.4.07": ("EXCAVACION POZO DE AVANCE EN TIERRA (INCL.DESALOJO A PUNTO A DE ACOPIO)", "m3"),
                "05.012.4.08": ("EXCAVACION POZO DE AVANCE EN CONGLOMERADO (INCL.DESALOJO A PUNTO A DE ACOPIO)", "m3"),
                "05.012.4.09": ("EXCAVACION POZO DE AVANCE EN ROCA (INCL.DESALOJO A PUNTO A DE ACOPIO)", "m3"),
                "01.016.4.09": ("MATERIAL DE MEJORAMIENTO", "m3"),
                "01.005.4.01": ("RELLENO COMPACTADO MATERIAL DE EXCAVACION  - EQUIPO LIVIANO", "m3"),
                "500327": ("REPLANTILLO DE PIEDRA, ESPESOR=20 CM", "m2"),
                "01.011.4.30": ("HORMIGON PREMEZCLADO REPLANTILLO FC=180 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "500409": ("ENTIBADO DE POZO METALICO (RETORNABLE)", "m2"),
                "01.007.4.80": ("TRANSPORTE MANUAL HORIZONTAL (CARRETILLA O SIMILAR) (SE PAGARÁ POR M3XM)", "u"),
                "01.007.4.02": ("ACARREO MECANICO HASTA 1 KM (CARGA,TRANSPORTE,VOLTEO)", "m3"),
                "01.007.4.63": ("SOBREACARREO MATERIAL NO UTILIZABLE A BOTADERO (TRANSPORTE/MEDIOS MECÁNICOS) (SE PAGARA EN M3-KM ) - NO INCL.CARGA", "u"),
            },

            "TUNEL": {
                "01.001.4.03": ("REPLANTEO Y NIVELACION TUNEL", "m"),
                "01.006.4.01": ("EXCAVACION TUNEL A MANO EN TIERRA (INC. DESALOJO HORIZONTAL Y VERTICAL)", "m3"),
                "01.006.4.04": ("EXCAVACION TUNEL A MANO EN ROCA -EQUIPO LIVIANO (INC. DESALOJO HORIZONTAL Y VERTICAL)", "m3"),
                "01.006.4.06": ("EXCAVACION TUNEL A MANO EN CONGLOMERADO - EQUIPO LIVIANO (INC. DESALOJO HORIZONTAL Y VERTICAL)", "m3"),
                "01.008.4.09": ("SOSTENIMIENTO TUNEL/ESTRUCTURA TIPO CELOSIA EN VARILLAS DE ACERO INCLUYE PLACA (NO RETORNABLE)", "Kg"),
                "01.008.4.20": ("ENTIBADO NO RETORNABLE EN TUNEL DE MADERA (PROVISION Y MONTAJE)", "m2"),
                "01.010.4.1": ("ENCOFRADO/DESENCOFRADO METALICO TUNEL/COLECTOR (BOVEDA-ARCO)", "m2"),
                "01.010.4.36": ("ENCOFRADO/DESENCOFRADO METALICO TUNEL/COLECTOR (RECTO)", "m2"),
                "01.011.4.30": ("HORMIGON PREMEZCLADO REPLANTILLO FC=180 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.011.4151": ("HORMIGON PREMEZCLADO FC=280 KG/CM2 TUNEL - A/C 0,45 - INCL ADITIVO, BOMBA Y TRANSPORTE", "m3"),
                "01.009.4.14": ("ACERO REFUERZO FY=4200 KG/CM2 TUNEL (SUMINISTRO, CORTE Y COLOCADO) INCL. ACARREO", "Kg"),
                "01.012.4.02": ("JUNTAS IMPERMEABLES PVC 15 CM (PROVISION Y MONTAJE)", "m"),
                "01.015.4.05": ("DRENES (TUBERIA PVC 110MM)", "m"),
                "01.007.4.80": ("TRANSPORTE MANUAL HORIZONTAL (CARRETILLA O SIMILAR) (SE PAGARÁ POR M3XM)", "u"),
                "01.007.4.02": ("ACARREO MECANICO HASTA 1 KM (CARGA,TRANSPORTE,VOLTEO)", "m3"),
                "01.007.4.63": ("SOBREACARREO MATERIAL NO UTILIZABLE A BOTADERO (TRANSPORTE/MEDIOS MECÁNICOS) (SE PAGARA EN M3-KM ) - NO INCL.CARGA", "u"),
            },

            "POZO DE AVANCE": {
                "05.012.4.07": ("EXCAVACION POZO DE AVANCE EN TIERRA (INCL.DESALOJO A PUNTO A DE ACOPIO)", "m3"),
                "05.012.4.08": ("EXCAVACION POZO DE AVANCE EN CONGLOMERADO (INCL.DESALOJO A PUNTO A DE ACOPIO)", "m3"),
                "05.012.4.09": ("EXCAVACION POZO DE AVANCE EN ROCA (INCL.DESALOJO A PUNTO A DE ACOPIO)", "m3"),
                "01.016.4.09": ("MATERIAL DE MEJORAMIENTO", "m3"),
                "01.005.4.01": ("RELLENO COMPACTADO MATERIAL DE EXCAVACION  - EQUIPO LIVIANO", "m3"),
                "500327": ("REPLANTILLO DE PIEDRA, ESPESOR=20 CM", "m2"),
                "01.011.4.30": ("HORMIGON PREMEZCLADO REPLANTILLO FC=180 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.011.4118": ("HORMIGON PREMEZCLADO FC=280 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.009.4.01": ("ACERO REFUERZO FY=4200 KG/CM2 (SUMINISTRO, CORTE Y COLOCADO)", "kg"),
                "01.010.4.13": ("ENCOFRADO/DESENCOFRADO METALICO RECTO", "m2"),
                "01.010.4.37": ("ENCOFRADO/DESENCOFRADO TABLERO CONTRACHAPADO", "m2"),
                "500409": ("ENTIBADO DE POZO METALICO (RETORNABLE)", "m2"),
                "500298": ("TAPA Y CERCO HIERRO DUCTIL D=850mm POZO REVISION ( GRUPO C - 40 Ton NORMA NTE INEN 2496 o GRUPO D 40 Ton NORMA NTE INEN EN124-1) (PROVISION Y MONTAJE)", "u"),
                "01.025.4.01": ("ESTRIBO DE VARILLA 16MM GALVANIZADO EN CALIENTE PARA PELDAÑO (PROVISION Y MONTAJE)", "u"),
                "01.007.4.80": ("TRANSPORTE MANUAL HORIZONTAL (CARRETILLA O SIMILAR) (SE PAGARÁ POR M3XM)", "u"),
                "01.007.4.02": ("ACARREO MECANICO HASTA 1 KM (CARGA,TRANSPORTE,VOLTEO)", "m3"),
                "01.007.4.63": ("SOBREACARREO MATERIAL NO UTILIZABLE A BOTADERO (TRANSPORTE/MEDIOS MECÁNICOS) (SE PAGARA EN M3-KM ) - NO INCL.CARGA", "u"),
            },

            "POZOS DE REVISION": {
                "03.007.4123": ("POZO REVISION H.S. TIPO B1 H=1.26-1.75M DIAM. EXT =1.40 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4124": ("POZO REVISION H.S. TIPO B1 H=1.76-2.25M DIAM. EXT =1.40 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4125": ("POZO REVISION H.S. TIPO B1 H=2.26-2.75M DIAM. EXT =1.40 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4126": ("POZO REVISION H.S. TIPO B1 H=2.76-3.25M DIAM. EXT =1.40 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4127": ("POZO REVISION H.S. TIPO B1 H=3.26-3.75M DIAM. EXT =1.40 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4128": ("POZO REVISION H.S. TIPO B1 H=3.76-4.25M DIAM. EXT =1.40 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4129": ("POZO REVISION H.S. TIPO B1 H=4.26-4.75M DIAM. EXT =1.40 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4130": ("POZO REVISION H.S. TIPO B1 H=4.76-5.25M DIAM. EXT =1.40 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4131": ("POZO REVISION H.S. TIPO B1 H=5.26-5.75M DIAM. EXT =1.40 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4132": ("POZO REVISION H.S. TIPO B2 H=2.51-2.99M DIAM. EXT =1.90 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4133": ("POZO REVISION H.S. TIPO B2 H=3.00-3.49M DIAM. EXT =1.90 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4134": ("POZO REVISION H.S. TIPO B2 H=3.50-3.99M DIAM. EXT =1.90 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4135": ("POZO REVISION H.S. TIPO B2 H=4.00-4.49M DIAM. EXT =1.90 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4136": ("POZO REVISION H.S. TIPO B2 H=4.50-4.99M DIAM. EXT =1.90 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4137": ("POZO REVISION H.S. TIPO B2 H=5.00-5.49M DIAM. EXT =1.90 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4138": ("POZO REVISION H.S. TIPO B2 H=5.50-6.00M DIAM. EXT =1.90 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4139": ("POZO REVISION H.S. TIPO B3 H=2.51-2.99M DIAM. EXT =2.20 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4140": ("POZO REVISION H.S. TIPO B3 H=3.00-3.49M DIAM. EXT =2.20 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4141": ("POZO REVISION H.S. TIPO B3 H=3.50-3.99M DIAM. EXT =2.20 M(TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4142": ("POZO REVISION H.S. TIPO B3 H=4.00-4.49M DIAM. EXT =2.20 M(TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4143": ("POZO REVISION H.S. TIPO B3 H=4.50-4.99M DIAM. EXT =2.20 M(TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4144": ("POZO REVISION H.S. TIPO B3 H=5.00-5.49M DIAM. EXT =2.20 M(TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "03.007.4145": ("POZO REVISION H.S. TIPO B3 H=5.50-6.00M DIAM. EXT =2.20 M(TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "500401": ("POZO REVISION H.S. TIPO B4 H=2.51-2.99M DIAM. EXT =2.50 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "500402": ("POZO REVISION H.S. TIPO B4 H=3.00-3.49M DIAM. EXT =2.50 M (TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "500403": ("POZO REVISION H.S. TIPO B4 H=3.50-3.99M DIAM. EXT =2.50 M(TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "500404": ("POZO REVISION H.S. TIPO B4 H=4.00-4.49M DIAM. EXT =2.50 M(TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "500405": ("POZO REVISION H.S. TIPO B4 H=4.50-4.99M DIAM. EXT =2.50 M(TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "500406": ("POZO REVISION H.S. TIPO B4 H=5.00-5.49M DIAM. EXT =2.50 M(TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
                "500407": ("POZO REVISION H.S. TIPO B4 H=5.50-6.00M DIAM. EXT =2.50 M(TAPA-CERCO H.DUCTIL ABISAGRADA CARGA DE ENSAYO 40 TON Y PELDAÑOS)", "u"),
            },

            "POZO SALTO TIPO S1": {
                "01.003.4.01": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (EN TIERRA)", "m3"),
                "01.003.4.02": ("EXCAVACION ZANJA A MANO H=2,76-3,99M (EN TIERRA)", "m3"),
                "01.003.4.03": ("EXCAVACION ZANJA A MANO H=4,00-6,00M (EN TIERRA)", "m3"),
                "01.003.4.27": ("EXCAVACION ZANJA A MAQUINA H>6,00M (EN TIERRA)", "m3"),
                "01.016.4.09": ("MATERIAL DE MEJORAMIENTO", "m3"),
                "01.005.4.01": ("RELLENO COMPACTADO MATERIAL DE EXCAVACION  - EQUIPO LIVIANO", "m3"),
                "500327": ("REPLANTILLO DE PIEDRA, ESPESOR=20 CM", "m2"),
                "01.011.4.30": ("HORMIGON PREMEZCLADO REPLANTILLO FC=180 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.011.4118": ("HORMIGON PREMEZCLADO FC=280 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.009.4.01": ("ACERO REFUERZO FY=4200 KG/CM2 (SUMINISTRO, CORTE Y COLOCADO)", "kg"),
                "01.010.4.13": ("ENCOFRADO/DESENCOFRADO METALICO RECTO", "m2"),
                "01.010.4.37": ("ENCOFRADO/DESENCOFRADO TABLERO CONTRACHAPADO", "m2"),
                "500409": ("ENTIBADO DE POZO METALICO (RETORNABLE)", "m2"),
                "500298": ("TAPA Y CERCO HIERRO DUCTIL D=850mm POZO REVISION ( GRUPO C - 40 Ton NORMA NTE INEN 2496 o GRUPO D 40 Ton NORMA NTE INEN EN124-1) (PROVISION Y MONTAJE)", "u"),
                "01.025.4.01": ("ESTRIBO DE VARILLA 16MM GALVANIZADO EN CALIENTE PARA PELDAÑO (PROVISION Y MONTAJE)", "u"),
                "01.007.4.80": ("TRANSPORTE MANUAL HORIZONTAL (CARRETILLA O SIMILAR) (SE PAGARÁ POR M3XM)", "u"),
                "01.007.4.02": ("ACARREO MECANICO HASTA 1 KM (CARGA,TRANSPORTE,VOLTEO)", "m3"),
                "01.007.4.63": ("SOBREACARREO MATERIAL NO UTILIZABLE A BOTADERO (TRANSPORTE/MEDIOS MECÁNICOS) (SE PAGARA EN M3-KM ) - NO INCL.CARGA", "u"),
            },

            "POZO SALTO TIPO S2": {
                "01.003.4.01": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (EN TIERRA)", "m3"),
                "01.003.4.02": ("EXCAVACION ZANJA A MANO H=2,76-3,99M (EN TIERRA)", "m3"),
                "01.003.4.03": ("EXCAVACION ZANJA A MANO H=4,00-6,00M (EN TIERRA)", "m3"),
                "01.003.4.27": ("EXCAVACION ZANJA A MAQUINA H>6,00M (EN TIERRA)", "m3"),
                "01.016.4.09": ("MATERIAL DE MEJORAMIENTO", "m3"),
                "01.005.4.01": ("RELLENO COMPACTADO MATERIAL DE EXCAVACION  - EQUIPO LIVIANO", "m3"),
                "500327": ("REPLANTILLO DE PIEDRA, ESPESOR=20 CM", "m2"),
                "01.011.4.30": ("HORMIGON PREMEZCLADO REPLANTILLO FC=180 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.011.4118": ("HORMIGON PREMEZCLADO FC=280 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.009.4.01": ("ACERO REFUERZO FY=4200 KG/CM2 (SUMINISTRO, CORTE Y COLOCADO)", "kg"),
                "01.010.4.13": ("ENCOFRADO/DESENCOFRADO METALICO RECTO", "m2"),
                "01.010.4.37": ("ENCOFRADO/DESENCOFRADO TABLERO CONTRACHAPADO", "m2"),
                "500409": ("ENTIBADO DE POZO METALICO (RETORNABLE)", "m2"),
                "01.011.4.08": ("HORMIGON CICLOPEO  40% PIEDRA (FC=210 KG/CM2) EN SITIO", "m3"),
                "01.009.4.83": ("MALLA ELECTROSOLDADA FY=5000KG/CM2 (PROVISION Y MONTAJE)", "kg"),
                "500298": ("TAPA Y CERCO HIERRO DUCTIL D=850mm POZO REVISION ( GRUPO C - 40 Ton NORMA NTE INEN 2496 o GRUPO D 40 Ton NORMA NTE INEN EN124-1) (PROVISION Y MONTAJE)", "u"),
                "01.025.4.01": ("ESTRIBO DE VARILLA 16MM GALVANIZADO EN CALIENTE PARA PELDAÑO (PROVISION Y MONTAJE)", "u"),

                '500497': ('TUBERIA ACERO ASTM A53 GRADO B RECUBIERTA 12" CD 40 (RECUBRIMIENTO INTERIOR Y EXTERIOR DE ACUERDO A NORMA AWWA C210)) (MAT/TRANS/COL)', 'm'),
                '500498': ('TUBERIA ACERO ASTM A53 GRADO B RECUBIERTA 16" CD 40 (RECUBRIMIENTO INTERIOR Y EXTERIOR DE ACUERDO A NORMA AWWA C210)) (MAT/TRANS/COL)', 'm'),
                '500499': ('TUBERIA ACERO ASTM A53 GRADO B RECUBIERTA 20" CD 40 (RECUBRIMIENTO INTERIOR Y EXTERIOR DE ACUERDO A NORMA AWWA C210)) (MAT/TRANS/COL)', 'm'),
                '500495': ('TUBERIA ACERO ASTM A53 GRADO B RECUBIERTA 24" CD 40 (RECUBRIMIENTO INTERIOR Y EXTERIOR DE ACUERDO A NORMA AWWA C210)) (MAT/TRANS/COL)', 'm'),
                '500257': ('UNION MECANICA LAMINA DE ACERO 12" (MAT/TRANS/INST)', 'u'),
                '500259': ('UNION MECANICA LAMINA DE ACERO 16" (MAT/TRANS/INST)', 'u'),
                '500260': ('UNION MECANICA LAMINA DE ACERO 20" (MAT/TRANS/INST)', 'u'),
                '500261': ('UNION MECANICA LAMINA DE ACERO 24" (MAT/TRANS/INST)', 'u'),
                '500501': ('ACCESORIO DE ACERO DE 12" (INCL. RECUBRIMIENTO DE ACUERDO A NORMA: INTERNO Y EXTERNO)', 'Kg'),
                '500502': ('ACCESORIO DE ACERO DE 16" (INCL. RECUBRIMIENTO DE ACUERDO A NORMA: INTERNO Y EXTERNO)', 'Kg'),
                '500503': ('ACCESORIO DE ACERO DE 20" (INCL. RECUBRIMIENTO DE ACUERDO A NORMA: INTERNO Y EXTERNO)', 'Kg'),
                '500504': ('ACCESORIO DE ACERO DE 24" (INCL. RECUBRIMIENTO DE ACUERDO A NORMA: INTERNO Y EXTERNO)', 'Kg'),

                "01.007.4.80": ("TRANSPORTE MANUAL HORIZONTAL (CARRETILLA O SIMILAR) (SE PAGARÁ POR M3XM)", "u"),
                "01.007.4.02": ("ACARREO MECANICO HASTA 1 KM (CARGA,TRANSPORTE,VOLTEO)", "m3"),
                "01.007.4.63": ("SOBREACARREO MATERIAL NO UTILIZABLE A BOTADERO (TRANSPORTE/MEDIOS MECÁNICOS) (SE PAGARA EN M3-KM ) - NO INCL.CARGA", "u"),

            },

            "REPOSICION DE LASTRE": {
                "535248": ("SUMINISTRO DE MATERIAL DE LASTRE (INCLUYE ESPONJAMIENTO) ", "m3"),
                "532003": ("HORA DE MAQUINA MOTONIVELDORA (INCLUYE OPERADOR) (SE PAGARA POR HORA)", "u"),
                "532004": ("HORA DE MAQUINA RODILLO COMPACTADOR (INCLUYE OPERADOR) (SE PAGARA POR HORA)", "u"),
                "532005": ("HORA DE TANQUERO DE AGUA (INCLUYE OPERADOR Y AGUA) (SE PAGARA POR HORA)", "u"),
            },

            "CORTE , ROTURA Y REPOSICION DE PAVIMENTO RIGIDO": {
                "01.011.4118": ("HORMIGON PREMEZCLADO FC=280 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.009.4.01": ("ACERO REFUERZO FY=4200 KG/CM2 (SUMINISTRO, CORTE Y COLOCADO)", "kg"),
                "01.016.4.82": ("ROTURA PAVIMENTO RIGIDO (PIONJER) (INCL. CORTE DE FILOS)", "m3"),
            },

            "CORTE , ROTURA Y REPOSICION DE PAVIMENTO FLEXIBLE": {
                "01.016.4.97": ("CARPETA ASFALTICA EN ZANJAS DE REDES (EQUIPO LIVIANO) - INCL. MATERIAL,TRANSPORTE,TENDIDO,COMPACTADO,IMPRIMACION", "m3"),
                "01.016.4.79": ("ROTURA DE CARPETA ASFALTICA INCLUYE ACOPIO LATERAL FUERA DE LA ZANJA (INCL. CORTE DE FILOS)", "m3"),
            },

            "CORTE , ROTURA Y REPOSICION DE VEREDAS": {
                "01.011.4.01": ("HORMIGON PREMEZCLADO FC=210 KG/CM2 INCLUYE ADITIVO, TRANSPORTE", "m3"),
                "01.009.4.83": ("MALLA ELECTROSOLDADA FY=5000KG/CM2 (PROVISION Y MONTAJE)", "kg"),
                "500306": ("BOMBEO DE AGUA (SE PAGARA POR HORA)", "m3"),
            },

            "REMOCION Y COLOCACION DE ADOQUIN": {
                "01.016.4.23": ("DESADOQUINADO", "m2"),
                "01.016.4.25": ("READOQUINADO (MATERIAL EXISTENTE) - INCL. CAMA DE ARENA Y EMPORADO", "m2"),
            },

            "DEROCAMIENTO": {
                "01.030.4.01": ("DERROCAMIENTO HORMIGON ARMADO (HERRAMIENTA MENOR / EQUIPO LIVIANO)", "m3"),
                "01.030.4.11": ("DERROCAMIENTO POZO HORMIGON SIMPLE (INCL. ELEVADOR)", "m3"),
                "01.030.4.10": ("DERROCAMIENTO POZO MAMPOSTERIA LADRILLO (INCL. ELEVADOR)", "m3"),
                "01.030.4.04": ("DERROCAMIENTO MAMPOSTERIA BLOQUE (HERRAMIENTA MENOR / EQUIPO LIVIANO)", "m3"),
            },

            "ADICIONALES": {
                "01.036.4100": ("OBJETO AS-BUILT SIG", "u"),
                "01.036.4.01": ("NIVELACION POZO A POZO PARA CATASTRO-INCLUYE CALCULO LIBRETA Y DIBUJO PERFILES", "km"),
                "01.031.4.04": ("INSPECCION CCTV CON CAMARA AUTOPROPULSADA INCLUYE INFORME", "m"),
            },

            "DOMICILIARIAS": {
                "01.003.4.01": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (EN TIERRA)", "m3"),
                "01.003.4.05": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (CONGLOMERADO)", "m3"),
                "01.003.4.13": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (ROCA) INCL. EQUIPO LIVIANO", "m3"),
                "01.005.4.01": ("RELLENO COMPACTADO MATERIAL DE EXCAVACION  - EQUIPO LIVIANO", "m3"),
                "01.016.4.09": ("MATERIAL DE MEJORAMIENTO", "m3"),
                "03.004.4197": ("TUBERIA PVC UE ALCANTARILLADO D.I.N. 160MM (MAT.TRAN.INST)", "m"),
                "500335": ("SUM. INST, POZO TIL PVC DN=160 MM PVC (INCLUYE TAPA)", "u"),
                "03.008.4.69": ("CAJA DOMICILIARIA HORMIGÓN 280 KG/CM2, H= MENOR O IGUAL A 1.50M, INCLUYE TAPA", "u"),
                "500342": ("SUM, INST ADAPTADOR PVC PARA ALCANT. A TUBERÍA D=160 MM", "u"),
                "01.007.4.80": ("TRANSPORTE MANUAL HORIZONTAL (CARRETILLA O SIMILAR) (SE PAGARÁ POR M3XM)", "u"),
                "01.007.4.02": ("ACARREO MECANICO HASTA 1 KM (CARGA,TRANSPORTE,VOLTEO)", "u"),
                "01.007.4.63": ("SOBREACARREO MATERIAL NO UTILIZABLE A BOTADERO (TRANSPORTE/MEDIOS MECÁNICOS) (SE PAGARA EN M3-KM ) - NO INCL.CARGA", "u"),
                "500343": ("CATASTRO DE DOMICILIARIAS EN SISTEMAS DE ALCANTARILLADO", "u"),
            },

            "SUMIDEROS": {
                "500345": ("SUM,-INS, REJILLA DE MATERIAL COMPUESTO PARA SUMIDERO 450 X 350MM - RESISTENICA 250KN", "u"),
                "500346": ("SUM, INST POZO PARA SUMIDERO PVC H=1.00 M (INCLUYE CAUCHOS DE EMPAQUE)", "u"),
                "03.004.4198": ("TUBERIA PVC UE ALCANTARILLADO D.I.N. 200MM (MAT.TRAN.INST)", "m"),
                "01.003.4.01": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (EN TIERRA)", "m3"),
                "01.003.4.05": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (CONGLOMERADO)", "m3"),
                "01.003.4.13": ("EXCAVACION ZANJA A MANO H=0,00-2,75M (ROCA) INCL. EQUIPO LIVIANO", "m3"),
                "01.005.4.01": ("RELLENO COMPACTADO MATERIAL DE EXCAVACION  - EQUIPO LIVIANO", "m3"),
                "01.016.4.09": ("MATERIAL DE MEJORAMIENTO", "m3"),
                "01.007.4.80": ("TRANSPORTE MANUAL HORIZONTAL (CARRETILLA O SIMILAR) (SE PAGARÁ POR M3XM)", "u"),
                "01.007.4.02": ("ACARREO MECANICO HASTA 1 KM (CARGA,TRANSPORTE,VOLTEO)", "m3"),
                "01.007.4.63": ("SOBREACARREO MATERIAL NO UTILIZABLE A BOTADERO (TRANSPORTE/MEDIOS MECÁNICOS) (SE PAGARA EN M3-KM ) - NO INCL.CARGA", "u"),
            },

            "PLAN DE MANEJO SOCIOAMBIENTAL": {
                "01.022.4.07": ("POLIETILENO 0,2 MM (INCLUYE INSTALACION)", "m2"),
                "01.024.4.01": ("ROTULO CON CARACTERISTICAS DEL PROYECTO (PROVISION Y MONTAJE)", "m2"),
                "01.024.4.08": ("CONO DE SEÑALIZACION VIAL (H MÍNIMA 90CM) (PROVISION Y COLOCACION)", "u"),
                "01.024.4.60": ("PROTECCIÓN COLECTIVA SEÑALÉTICA DE CIERRE DE TRÁNSITO/PASO PEATONAL-POSTE H MÍNIMO 1M - (1.20X0.75M) TOOL Y LAMINA REFLECTIVA (NORMA ASTM D4956) (UNIDAD POR MES)", "u"),
                "01.024.4.61": ("PROTECCIÓN COLECTIVA SEÑALÉTICA DE CIERRE PARCIAL DE VIA- POSTE H MÍNIMO 1M - (0.75X0.75M) TOOL Y LAMINA REFLECTIVA (NORMA ASTM D4956) (UNIDAD POR MES)", "u"),
                "01.024.4.67": ("PROTECCIÓN COLECTIVA CON BARRICADA TIPO III (H MINIMA=1.50M L MÍNIMA=1.20 M) (NORMA ASTM D4956) (UNIDAD POR MES)", "u"),
                "01.024.4.72": ("PROTECCIÓN COLECTIVA FIJA CON CANALIZADORES TIPO NEW JERSEY (ALTURA MÍNIMA =0.7 M ; LONGITUD MÍNIMA=1.20 M) (PROVISION Y MONTAJE)(UNIDAD POR MES)", "u"),
                "03.016.4.01": ("PASO PEATONAL DE MADERA 1.2M ANCHO (VARIOS USOS)", "m"),
                "04.020.4.55": ("CERRAMIENTO PROVISIONAL DE TOOL,PINGO/VIGA (SUMINISTRO, MONTAJE Y PINTURA)", "m2"),
                "04.020.4.74": ("CUBIERTA PROVISIONAL DE TOOL,PINGO/VIGA (SUMINISTRO, MONTAJE Y PINTURA)", "m2"),
                "07.001.4.05": ("CONTROL DE POLVO (INCL. AGUA Y TANQUERO)", "m3"),
                # "500475": ("LEVANTAMIENTO DE FICHA DE INSPECCION PREVIA A VIVIENDAS", "u"),
            },

            "REPARACION Y REPOSICION DOMICILIARIAS DE AGUA POTABLE": {
                "02.024.4409": ("TUBERIA DE COBRE FLEXIBLE TIPO K 1/2 (PROVISION Y INSTALACION)", "m"),
                "02.024.4572": ("UNION DE DOS PARTES 1/2 (NORMA AWWA C800-21) PARA TUBERIA DE COBRE (PROVISION Y MONTAJE)", "u"),
                "02.024.4740": ("TOMA DE INCORPORACION 1/2 A 1/2 (NORMA AWWA C800-21) INCLUYE UNION A TUBERIA DE COBRE 1/2 (PROVISION Y MONTAJE)", "u"),
                "02.024.4742": ("VALVULA DE COMPUERTA 1/2 (NPT H-H) (NORMA AWWA C800-21) (PROVISION Y MONTAJE)", "u"),
                "02.024.4753": ("CODO 90° ACERO INOX. 1/2 (NORMA AISI 304) ROSCA NPT (PROVISION E INSTALACION)", "u"),
                "02.024.4755": ("NEPLO CON TUERCA/NEPLO CORRIDO MACHO ACERO INOX. 1/2 (NORMA AISI 304) ROSCA NPT (PROVISION E INSTALACION)", "u"),
                "02.024.4782": ("UNION UNIVERSAL ACERO INOX. 1/2 (NORMA AISI 304) ROSCA NPT (PROVISION E INSTALACION)", "u"),
                "02.024.4715": ("COLLAR H.DUCTIL Y/O ACERO INOX. - AWWA C800-21 (CONEX. AGUA POTABLE) 63MMX1/2 (PROVISION Y MONTAJE)", "u"),
                "02.024.4718": ("COLLAR H.DUCTIL Y/O ACERO INOX. - AWWA C800-21 (CONEX. AGUA POTABLE) 90MMX1/2 (PROVISION Y MONTAJE)", "u"),
                "02.024.4721": ("COLLAR H.DUCTIL Y/O ACERO INOX. - AWWA C800-21 (CONEX. AGUA POTABLE) 110MMX1/2 (PROVISION Y MONTAJE)", "u"),
                "02.024.4724": ("COLLAR H.DUCTIL Y/O ACERO INOX. - AWWA C800-21 (CONEX. AGUA POTABLE) 160MMX1/2 (PROVISION Y MONTAJE)", "u"),
            },

            "REPARACION Y REPOSICION TUBERIA DE ACERO DE AGUA POTABLE": {
                "02.002.4198": ("TUBERIA ACERO ASTM A53 GRADO B RECUBIERTA 02 CD 40 (RECUBRIMIENTO INTERIOR Y EXTERIOR DE ACUERDO A NORMA AWWA C210) (MAT/TRANS/COL)", "m"),
                "02.002.4199": ("TUBERIA ACERO ASTM A53 GRADO B RECUBIERTA 03 CD 40 (RECUBRIMIENTO INTERIOR Y EXTERIOR DE ACUERDO A NORMA AWWA C210) (MAT/TRANS/COL)", "m"),
                "02.002.4200": ("TUBERIA ACERO ASTM A53 GRADO B RECUBIERTA 04 CD 40 (RECUBRIMIENTO INTERIOR Y EXTERIOR DE ACUERDO A NORMA AWWA C210)) (MAT/TRANS/COL)", "m"),
                "02.002.4201": ("TUBERIA ACERO ASTM A53 GRADO B RECUBIERTA 06 CD 40 (RECUBRIMIENTO INTERIOR Y EXTERIOR DE ACUERDO A NORMA AWWA C210)) (MAT/TRANS/COL)", "m"),
                "02.018.4.65": ("UNION MECANICA LAMINA DE ACERO 02 (MAT/TRANS/INST)", "u"),
                "02.018.4.66": ("UNION MECANICA LAMINA DE ACERO 03 (MAT/TRANS/INST)", "u"),
                "02.018.4.67": ("UNION MECANICA LAMINA DE ACERO 04 (MAT/TRANS/INST)", "u"),
                "02.018.4.68": ("UNION MECANICA LAMINA DE ACERO 06 (MAT/TRANS/INST)", "u"),
                "02.004.5272": ("ACCESORIO DE ACERO DE 2 (INCL. RECUBRIMIENTO DE ACUERDO A NORMA: INTERNO Y EXTERNO)", "Kg"),
                "02.004.5273": ("ACCESORIO DE ACERO DE 3 (INCL. RECUBRIMIENTO DE ACUERDO A NORMA: INTERNO Y EXTERNO)", "Kg"),
                "02.004.5274": ("ACCESORIO DE ACERO DE 4 (INCL. RECUBRIMIENTO DE ACUERDO A NORMA: INTERNO Y EXTERNO)", "Kg"),
                "02.004.5275": ("ACCESORIO DE ACERO DE 6 (INCL. RECUBRIMIENTO DE ACUERDO A NORMA: INTERNO Y EXTERNO)", "Kg"),
                "02.026.4.16": ("CORDON DE SUELDA ELECTRICA EN TUBERIA ACERO E=12MM EN CAMPO - SOLDADOR API / ASME", "m"),
                "02.026.4.14": ("CORDON DE SUELDA ELECTRICA EN TUBERIA ACERO E=9MM EN CAMPO - SOLDADOR API / ASME", "m"),
                "02.026.4.05": ("CORTE TUBERIA ACERO EN CAMPO", "m"),
            },

            "TUBERIAS DE DESVIO TEMPORAL": {
                "01.020.4.21": ("DESVIO TUBERIA PLASTICA 200MM (MATERIAL VARIOS USOS)", "m"),
                "01.020.4.05": ("DESVIO TUBERIA PLASTICA 300MM (MATERIAL VARIOS USOS)", "m"),
                "01.020.4.06": ("DESVIO TUBERIA PLASTICA 600 MM (MATERIAL VARIOS USOS)", "m"),
                "01.020.4.17": ("DESVIO TUBERIA PLASTICA 900 MM (MATERIAL VARIOS USOS)", "m"),
            }
        }
        self.total_budget = 0
        self.df_presupuesto = None
        self.formatter = None
    
    def run(self, excel_output_path: str = None, excel_metadata: dict = None) -> float:
        """
        Ejecuta el flujo completo de cálculo de costos.
        Si fase es None, itera sobre todas las fases del GPKG.
        
        Args:
            excel_output_path (str, optional): Path to save the Excel budget.
            excel_metadata (dict, optional): Metadata for the Excel header (proyecto_name, sistema, etc).

        Returns:
            float: Presupuesto total calculado (suma de todas las fases si fase=None)
        """
        # Si fase es None, leer todas las fases del GPKG
        if self.fase is None:
            _df = gpd.read_file(self.vector_path)
            fases = _df['Fase'].unique()
        else:
            fases = [self.fase]
        
        self.results = {}  # Guardar resultados por fase
        total_all = 0
        
        for fase in fases:
            # Actualizar fase en parameters_dict
            self.parameters_dict['filtro'] = {'column': 'Fase', 'value': fase}
            current_fase = fase
            
            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            c_cr = CantidadesZanjaAbierta(self.parameters_dict)

            if not  c_cr.m_ramales_df.empty:
                # cantidades de excavacion
                df_cut_vol = c_cr.get_class_soil_vol_quantities(force_general_percentages=False)

                # cantidades de entibado
                df_shoring = c_cr.get_shoring()

                # cantidades de relleno
                df_fill, df_rotura = c_cr.get_fill_vol()

                # cantidades de mejoramiento de fondo de zanja
                df_soil_replacement = c_cr.get_soil_improvement()

                # cantidades acarreo y desalojo
                df_vol = pd.concat([df_cut_vol, df_soil_replacement])
                df_material_removal = c_cr.get_material_removal(df_vol)

                # abatimiento, desbroce
                df_other_quantities = c_cr.get_other_quantities()

                area_trench_properties = c_cr.area_properties

                # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                tp = CantidadesTuberiasPozos(c_cr.m_ramales_df, area_trench_properties, self.parameters_dict)

                resumen = tp.get_length_metodo_constructivo()

                # cantidades tuberia
                df_pipes = tp.get_pipe_length_circular()

                # cantidades canales y colectores
                df_canales_colectores = tp.get_pipe_length_rectangular()

                # cantidades pozos
                df_pz = tp.get_pz_class()
                cantidad_pozos = df_pz['CANTIDAD'].sum()

                # cantidades pozos especiales y salto
                df_pz_especial = tp.get_pozo_especial(c_cr.depth_ranges)
            else:
                cantidad_pozos = 1
                df_cut_vol = pd.DataFrame()
                df_shoring = pd.DataFrame()
                df_fill = pd.DataFrame()
                df_rotura = pd.DataFrame()
                df_soil_replacement = pd.DataFrame()
                df_material_removal = pd.DataFrame()
                df_other_quantities = pd.DataFrame()
                df_pipes = pd.DataFrame()
                df_canales_colectores = pd.DataFrame()
                df_pz = pd.DataFrame()
                df_pz_especial = pd.DataFrame()

            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            ct = CantidadesTunel(self.parameters_dict)
            df_tunel = ct.get_tunel_pozo_vol()

            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            phd = CantidadesPerforacionHorizontalDirigida(self.parameters_dict)
            df_phd = phd.get_phd()

            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            dm = CantidadesDerrocamiento(self.parameters_dict)
            df_derrocamiento = dm.get_derrocamiento_vol()

            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            ds = CantidadesDomiciliariasSumideros(self.parameters_dict, self.tipo, current_fase)
            df_domiciliarias = ds.get_domiciliarias()
            df_sumideros = ds.get_sumideros()
            if len(df_domiciliarias) > 0:
                numero_domiciliarias = df_domiciliarias.loc['catastro_domiciliaria']['CANTIDAD']
            else:
                numero_domiciliarias = 1

            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            ca = CantidadesAdicionales(self.parameters_dict, cantidad_pozos)
            df_cctv = ca.get_CCTV()
            df_sa = ca.get_socio_ambiental(numero_domiciliarias)
            df_aq = ca.get_catastro()

            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            crap = CantidadesReparacionesAguaPotable_DesvioAlcantarillado(self.parameters_dict, numero_domiciliarias)
            df_desvios = crap.get_quantities_tuberias_desvio()
            
            df_reparacion_tuberia_acero = pd.Series()
            df_reparacion_domiciliaria = pd.Series()

            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            df = pd.concat([df_cut_vol, df_shoring, df_fill, df_rotura, df_soil_replacement, df_material_removal, 
                           df_other_quantities, df_pipes, df_canales_colectores, df_pz, df_pz_especial, 
                           df_tunel, df_phd, df_derrocamiento, df_cctv, df_sa, df_aq, df_domiciliarias, 
                           df_sumideros, df_reparacion_tuberia_acero, df_reparacion_domiciliaria, df_desvios])
            df = df[df['CANTIDAD'] > 0.01]
            df = df.round(2)

            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            formatter = ExcelFormatter(self.data, self.tipo, current_fase, self.parameters_dict)

            # Create budget data using the results from CantidadesZanjaAbierta
            formatter.create_budget_data(vg.rubros_presupuesto, df)

            # Process the data
            formatter.flatten_data()

            # create dataframe with values on quantity
            formatter.add_budget_values()

            # Guardar resultados de esta fase
            self.results[current_fase] = {
                'total_budget': formatter.total_budget,
                'df_presupuesto': formatter.df_presupuesto,
                'formatter': formatter
            }
            total_all += formatter.total_budget
            

        
        # Si solo una fase, guardar referencia directa
        if len(fases) == 1:
            self.formatter = self.results[fases[0]]['formatter']
            self.total_budget = self.results[fases[0]]['total_budget']
            self.df_presupuesto = self.results[fases[0]]['df_presupuesto']
        else:
            self.total_budget = total_all
            self.formatter = None  # No hay un único formatter cuando hay múltiples fases
            
        # Export logic
        if excel_output_path and self.formatter:
             meta = excel_metadata or {}
             try:
                 print(f"  [SewerConstructionCost] Exporting Excel to: {excel_output_path}")
                 self.to_excel(
                     excel_output_path, 
                     proyecto_name=meta.get('proyecto_name', 'PROYECTO'),
                     sistema=meta.get('sistema', 'SISTEMA'),
                     ubicacion=meta.get('ubicacion', 'UBICACION'),
                     obra=meta.get('obra', 'OBRA'),
                     cliente=meta.get('cliente', 'CLIENTE')
                 )
             except Exception as e:
                 print(f"  [Error] Failed to export Excel: {e}")

        return self.total_budget
    
    def to_excel(self, output_path: str, proyecto_name: str, sistema: str, ubicacion: str, obra: str, cliente: str):
        """
        Exporta el presupuesto a un archivo Excel.
        """
        if self.formatter is None:
            raise ValueError("Debe ejecutar run() antes de exportar a Excel")
        
        self.formatter.to_excel(output_path, proyecto_name, sistema, ubicacion, obra, cliente)







if __name__ == '__main__':
    # Ejemplo de uso de la clase SewerConstructionCost
    vector_path = r'C:\Users\Alienware\OneDrive\SANTA_ISABEL\00_tanque_tormenta\codigos\optimization_results\Seq_Iter_01\Seq_Iter_01.gpkg'
    # vector_path = r'Casima.gpkg'
    domiciliarias_vector_path = None  # Opcional
    tipo = 'PLUVIAL'
    base_precios = r'base_precios.xlsx'
    
    info_dict = {
        'proyecto_name': 'nombre_proyecto',
        'sistema': 'nombre_sistema',
        'ubicacion': 'ubicacion',
        'obra': 'obra',
        'cliente': 'cliente'
    }
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OPCION 1: Procesar todas las fases automáticamente (fase=None)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    calculator = SewerConstructionCost(
        vector_path=vector_path,
        tipo=tipo,
        fase=None,  # None = procesa TODAS las fases del GPKG
        domiciliarias_vector_path=domiciliarias_vector_path,
        base_precios=base_precios,

    )
    
    # Ejecutar (el for loop está dentro de run())
    total = calculator.run()
    print(f"\nPresupuesto Total (todas las fases): ${total:,.2f}")
    
    # Exportar cada fase a Excel
    name_xlsx = os.path.splitext(os.path.basename(vector_path))[0]
    for fase, result in calculator.results.items():
        formatter = result['formatter']
        formatter.to_excel(
            f'fase{fase}_{name_xlsx}.xlsx',
            proyecto_name=info_dict['proyecto_name'].upper(),
            sistema=info_dict['sistema'].upper(),
            ubicacion=info_dict['ubicacion'].upper(),
            obra=f'ALCANTARILLADO {tipo} - COMPONENTE {fase}',
            cliente=info_dict['cliente'],
        )

