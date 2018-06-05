# Python script para baixar informações da API Dados Abertos
import urllib.request, json, csv

# Definindo dialetos
csv.register_dialect("vertical_bar", delimiter='|', quoting=csv.QUOTE_MINIMAL)

# Caminhos dos arquivos onde estão a lista de ID's e onde será gravado o texto das ementas
path_cod_proposicoes = r'C:\Users\TCC_ADS\Documents\TCC1\Random Tests\API Downloader\cod_proposicoes_origem.txt'
path_proposicoes = r'C:\Users\TCC_ADS\Documents\TCC1\Random Tests\API Downloader\proposicoes.txt'
url_dados_abertos_proposicoes = "https://dadosabertos.camara.leg.br/api/v2/proposicoes/"

# Abre o arquivo para gravar os dados
proposicoes = open(path_proposicoes,'w', newline='')

# Lê o CSV com os IDs das proposições
with open(path_cod_proposicoes) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        lista_id_proposicao = row
        # Acessa e lê API do Dados Abertos
        for id_proposicao in lista_id_proposicao:
            with urllib.request.urlopen(url_dados_abertos_proposicoes+id_proposicao) as url:
                # Decodifica o JSON recebido e grava nas respectivas variáveis os dados do número do id e texto da ementa
                data = json.loads(url.read().decode())
                id_prop = data['dados']['id']
                ementa = data['dados']['ementa']
                # Grava no arquivo as informações coletadas em formato CSV com delimitador de barra vertical
                writer = csv.writer(proposicoes, dialect="vertical_bar")
                writer.writerow((str(id_prop), str(ementa)))

