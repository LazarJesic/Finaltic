import base64
import io


def load_stocks(file):
    #"""#reading company names from the text file containing list of available """
    stock_names=[]
    with open(file) as f:
        orgs = f.readlines()
    
    for i in range(len(orgs)):
        stock_names.append(orgs[i].rstrip('\n').split(','))
    stock = [item for sublist in stock_names for item in sublist]
    f.close()
    return stock

def makePlotPath(plt):
        buf = io.BytesIO()
        plt = plt.get_figure()
        plt.savefig(buf, format="png")
        # Embed the result in the html output.
        plt = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt = f'data:image/png;base64,{plt}'
        return plt