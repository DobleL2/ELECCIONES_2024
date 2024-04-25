from fpdf import FPDF 
from tabulate import tabulate
import pandas as pd

def add_dataframe_to_pdf(pdf, dataframe, x, y):
    # Convertir el DataFrame a una cadena de texto formateada como una tabla
    table = tabulate(dataframe, headers='keys', tablefmt='simple')
    
    # Dividir la tabla en filas para agregarlas al PDF
    rows = table.split('\n')
    
    # Agregar cada fila al PDF
    for index, row in dataframe.iterrows():
        pdf.set_xy(x, y)
        pdf.set_font('Helvetica', '', 10) 
        pdf.cell(0, 0, row['First name']+'\t'+str(row['Age'])+'\t'+row['City'] )
        y += 5  # Ajustar la coordenada y para la siguiente fila


class PDFWithBackground(FPDF):
    def __init__(self):
        super().__init__()
        self.background = None

    def set_background(self, image_path):
        self.background = image_path

    def add_page(self, orientation=''):
        super().add_page(orientation)
        if self.background:
            self.image(self.background, 0, 0, self.w, self.h)

    def footer(self):
        # Posición a 1.5 cm desde el fondo
        self.set_y(-15)
        # Configurar la fuente para el pie de página
        self.set_font('Helvetica', 'I', 8)
        # Número de página
        self.cell(0, 10, 'Página ' + str(self.page_no()), 0, 0, 'C')
        
pdf = PDFWithBackground()
pdf.set_background('images/portada.png')
pdf.add_page()

jurisdiccion = 'NACIONAL'
pregunta = 'A (Resumen pregunta A)'
# Sample DataFrame
df_si = pd.DataFrame({
    "First name": ["Jules", "Mary", "Carlson", "Lucas"],
    "Last name": ["Smith", "Ramos", "Banks", "Cimon"],
    "Age": ["34", "45", "19", "31"],
    "City": ["San Juan", "Orlando", "Los Angeles", "Saint-Mathurin-sur-Loire"]
})
df_no = df_si.copy()


pdf.set_background('images/background.jpeg')
pdf.add_page()

pdf.set_xy(15,15)
pdf.add_font('Anton-Regular', '', 'fonts/Anton-Regular.ttf', uni=True)  # Register the custom font
pdf.set_font('Helvetica','B',size=30)   # Helvetica, Times, Courier
pdf.cell(0,0,'Resultados Conteo Rápido',0,1,'L')

pdf.set_xy(15,24)
pdf.set_font('Helvetica', '', 15)  # Use the registered font
pdf.cell(0,0,'Muestra Matemática',0,1,'L')

pdf.set_xy(15,35)
pdf.set_font('Helvetica', 'B', 18)  # Use the registered font
pdf.cell(0,0,'Jurisdicción:',0,1,'L')

pdf.set_xy(56,35)
pdf.set_font('Helvetica', '', 16)  # Use the registered font
pdf.cell(0,0,jurisdiccion,0,1,'L')

pdf.set_xy(15,43)
pdf.set_font('Helvetica', 'B', 18)  # Use the registered font
pdf.cell(0,0,'Pregunta:',0,1,'L')

pdf.set_xy(47,43)
pdf.set_font('Helvetica', '', 16)  # Use the registered font
pdf.cell(0,0,pregunta,0,1,'L')

image_si = 'test1'
imagen = f'images/tendencia/{image_si}.png'
pdf.image(f'{imagen}',x=15,y=55,w=145,h=65)

pdf.set_xy(160,65)
pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
pdf.cell(0,0,'Límite',0,1,'L')
pdf.set_xy(160,70)
pdf.cell(0,0,'superior',0,1,'L')

pdf.set_xy(160,83)
pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
pdf.cell(0,0,'Valor',0,1,'L')

pdf.set_xy(160,96)
pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
pdf.cell(0,0,'Límite',0,1,'L')
pdf.set_xy(160,101)
pdf.cell(0,0,'inferior',0,1,'L')

#Valores
lim_inf_si = '25.34 %'
pdf.set_xy(180,67)
pdf.set_font('Helvetica',size= 11)  # Use the registered font
pdf.cell(0,0,f'{lim_inf_si}',0,1,'L')

valor_si = '25.34 %'
pdf.set_xy(180,83)
pdf.set_font('Helvetica',size= 11)  # Use the registered font
pdf.cell(0,0,f'{valor_si}',0,1,'L')

lim_sup_si = '25.34 %'
pdf.set_xy(180,98)
pdf.set_font('Helvetica',size= 11)  # Use the registered font
pdf.cell(0,0,f'{lim_sup_si}',0,1,'L')


image_no = 'test2'
imagen = f'images/tendencia/{image_no}.png'
pdf.image(f'{imagen}',x=15,y=125,w=145,h=65)

pdf.set_xy(160,125+10)
pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
pdf.cell(0,0,'Límite',0,1,'L')
pdf.set_xy(160,130+10)
pdf.cell(0,0,'superior',0,1,'L')

pdf.set_xy(160,143+10)
pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
pdf.cell(0,0,'Valor',0,1,'L')

pdf.set_xy(160,156+10)
pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
pdf.cell(0,0,'Límite',0,1,'L')
pdf.set_xy(160,161+10)
pdf.cell(0,0,'inferior',0,1,'L')

#Valores
lim_inf_no = '25.34 %'
pdf.set_xy(180,67+70)
pdf.set_font('Helvetica',size= 11)  # Use the registered font
pdf.cell(0,0,f'{lim_inf_no}',0,1,'L')

valor_no = '25.34 %'
pdf.set_xy(180,83+70)
pdf.set_font('Helvetica',size= 11)  # Use the registered font
pdf.cell(0,0,f'{valor_no}',0,1,'L')

lim_sup_no = '25.34 %'
pdf.set_xy(180,98+70)
pdf.set_font('Helvetica',size= 11)  # Use the registered font
pdf.cell(0,0,f'{lim_sup_no}',0,1,'L')



pdf.set_xy(45,200)
pdf.set_font('Helvetica', 'B', 10)  # Use the registered font
pdf.cell(0,0,'Evolución tendencia SI',0,1,'L')

pdf.set_xy(130,200)
pdf.set_font('Helvetica', 'B', 10)  # Use the registered font
pdf.cell(0,0,'Evolución tendencia NO',0,1,'L')

def create_table(df,x,y):
    # Convert DataFrame to list of tuples
    pdf.set_xy(x,y)
    pdf.set_font("Helvetica", size=6)
    table_data = [df.columns.tolist()] + df.values.tolist()

    with pdf.table(width=75, align='L') as table:
        for data_row in table_data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)

create_table(df_si,25,205)
create_table(df_no,110,205)

pdf.output('reporte_orig.pdf')
