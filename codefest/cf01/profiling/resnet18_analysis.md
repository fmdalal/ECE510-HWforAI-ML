Identify the five layers with the highest MAC count.

| Layer Number | Layer Name     | MACs        | Parameter Counts | Location                |
|--------------|----------------|-------------|------------------|-------------------------|
| 1            | conv2d (conv1) | 118,013,952 | 9,408            | Input Layer (stem)      |
| 2            | conv2d (conv1) | 115,605,504 | 36,864           | Layer 1, BasicBlock (0) |
| 3            | conv2d (conv2) | 115,605,504 | 36,864           | Layer 1, BasicBlock (0) |
| 4            | conv2d (conv1) | 115,605,504 | 36,864           | Layer 1, BasicBlock (1) |
| 5            | conv2d (conv2) | 115,605,504 | 36,864           | Layer 1, BasicBlock (1) |

---

For the single most MAC-intensive layer, compute its arithmetic intensity.

The first layer is the most MAC-intensive. Below are the details
| Property for MAC Intensive Layer  | Values            |
|-----------------------------------|-------------------|
| Layer Name                        | conv2d (conv1)    |
| MACs                              | 118,013,952       |
| Parameter Counts                  | 9,408             |
| Input Shape                       | [1, 3,  224, 224] |
| Output Shape                      | [1, 64, 112, 112] |

Arithmetic intensity 	= (2 * MAC) / (weight bytes + activation bytes)

Weight (in bytes) 		= MAC * 4

						= 118,013,952 * 4
						
						= 472,055,808
						
Activation (in bytes)	= (Number of inputs + Number of neurons) * 4bytes

						= ((3 * 224 * 224) + (64 * 112 * 112)) * 4
						
						= (150,528 + 802,816) * 4
						
						= 3,813,376
						
Arithmetic intensity 	= (2 * 118,013,952) / (472,055,808 + 3,813,376)

						= 236,027,904 / 475,869,184
						
						= 0.496
						
