import torch
from char_embeddings import *
import numpy as np
from gan_files.GAN import *
import pickle

embedding = CharEmbeddings()

model = Vanilla_Text_Encoder(
            batch_size = 32,
            embedding = embedding.embedding,
            cell_size = 512,
            num_layers = 1,
            bidirectional=False,
            GPU=False,
            gpu_nummber =0,
            batch_first=True,
            dropout_probability=0.5
            )
print(model)
model.load_state_dict(torch.load("/Users/sudarshinityagi/PycharmProjects/text_rep_stac_gan/models/char_text_rep_512__40"))
model.eval()

rnn = model.rnn
# captions = ["A young girl riding a large dog in a open field.",
#  "A German Shepard dog catching a frisbee in its mouth.",
#  "A man standing on top of a paddle board with a dog on it.",
#  "a person pets a dog who is laying on the floor ",
#  "a dot that is looking out of the window",
#  "A man and two dogs are in a backyard.",
#  "many people and dogs in a field ",
#  "Two small  dogs are looking at the camera.",
#  "A large breed dog with black and brown fur is standing on a white and black floor.",
#  "a close up of a car mirror with a dog ",
#  "a dog is pulling on a frisbee with its owner",
#  "A man jumping his bicycle onto a bench.",
#  "A dog that is eating some food with a human.",
#  "People are standing in the grass flying kites.",
#  "A man sitting on top of a couch in a living room.",
#  "A large brown and white dog sitting before a boy eating food.",
#  "A dog that is looking at a can of food.",
#  "A dog sitting under a walkway next to a lush green field.",
#  "A dog is resting peacefully beside a television remote.",
#  "Busy street at night with cars and pedestrians. ",
#  "A little puppy on the leg of a large teddy bear.",
#  "A large brown and white dog laying on top of a red bed.",
#  "a close up of a person laying on a couch with a dog",
#  "A dog with lying on a couch near someone using a laptop.",
#  "The black dog is looking up while standing on a checkered tile floor.",
#  "A dog looking intently at something by a persons laptop.",
#  "A man sitting in a  brown chair using a laptop computer.",
#  "A picture of a scene in the streets.",
#  "A skate park next to a body of water and green park.",
#  "A little dog with a shirt inside a car looking up.",
#  "a dog outside looking through a screen door",
#  "a living room with a couch and television inside of it "]

captions = ["a bird that has landed on top of another bird",
"a bird flies over a large body of water.",
"a yellow and green object with a brown bird perched on top of it.",
"a bird with a berry in its mouth sits on a rock",
"a lot if birds are standing at the beach shore",
"a bird that is sitting on top of a tree.",
"a bird themed clock sitting inside of a green box.",
"a small white bird flying over the water.",
"a blue bird is sitting on a wooden post.",
"a cat and a bird are in front of a door that has garbage by it.",
"a white bird walking through a shallow area of water.",
"a bird sits on a rear view mirror",
"three tropical bird perched on top of high bare branches",
"a grey bird on beach with water and pier in the background.",
"a bird perched on a metal bar next to the sea",
"black birds on a wood fence near the ocean.",
"a bird with a foot on the top of a pole",
"a colorful bird is perched on a branch.",
"a dog walking down the street with a bunch of birds flying around.",
"a small brown bird is sitting inside by a window.",
"a gray bird about to perch on a branch",
"a black bird landing on the leaves in the water",
"several birds overlook the skyline of a distant city.",
"a close up of a small bird on a tree branch",
"a clock tower with a a black bird flying by it.",
"a bird is flying away from the person",
"black birds picking berries out of the tree",
"a bird trying to eat out of a bird house ",
"a bird standing in the sand and another bird landing on a blanket in the sand.",
"a person walks along a beach as some bird play in the water",
"a lone, blue and orange bird sits on a bare tree.",
"a giraffe that has some birds perched on it."
]

true_caption_tokens = np.array([embedding.convert_to_tokens(caption, 65)[0] for caption in captions],
                              dtype=np.long).reshape(32,67)

hidden_state = torch.FloatTensor(32, 512)
cell_state = torch.FloatTensor(32, 512)
hidden_state.data.normal_(0, 1)
cell_state.data.normal_(0, 1)
hidden_state = hidden_state.unsqueeze(0)
cell_state = cell_state.unsqueeze(0)

_, (true_hidden_state, true_cell_state) = rnn(hidden_state, cell_state, torch.from_numpy(true_caption_tokens))



true_encoded = torch.cat((true_hidden_state, true_cell_state), 0)
true_encoded = true_encoded.view(32, -1)
print(true_encoded.shape)

fileObject = open("encoded_captions_bird.pkl",'wb')
pickle.dump(true_encoded.data.numpy(),fileObject, protocol=2)
fileObject.close()



