
from gensim.models import Word2Vec
textread= open("/Users/eunseo/Desktop/catall1910", "r")
text = textread.read()
textinput = text.split()
model = Word2Vec([textinput], size=300, window=3, min_count=3, workers=4)
model.save("/Users/eunseo/concept_vectors/model1")


