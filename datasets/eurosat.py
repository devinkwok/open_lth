# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform


class Dataset(base.ImageDataset, base.NdarrayDataset):
    """The EuroSAT dataset.
    Default test set consists of hardcoded set of 500 images per class.
    (note, these images are 4x the size of CIFAR-10 images, so the scale of the train and test sets are similar.
    """
    TEST_SPLIT = [0,14,29,30,32,43,44,45,51,52,56,63,67,70,73,80,87,88,93,102,108,111,120,124,134,135,139,141,144,149,152,162,170,174,175,177,178,183,184,188,192,194,196,203,211,218,233,239,246,251,254,256,257,266,270,282,289,291,296,298,309,313,314,318,321,322,324,331,332,343,354,366,368,387,393,402,408,410,411,414,416,420,423,430,432,433,436,439,443,450,457,460,463,471,472,478,479,485,486,495,506,507,511,521,528,532,533,535,543,544,554,555,557,564,567,568,572,594,598,602,605,612,621,644,650,651,679,680,685,693,695,705,718,727,729,741,746,755,759,761,764,765,772,781,783,785,787,789,794,798,807,809,817,831,840,842,844,845,857,862,864,874,879,881,897,900,903,907,911,912,927,929,930,942,945,965,978,998,1001,1005,1025,1027,1034,1041,1044,1047,1055,1057,1064,1073,1078,1080,1084,1090,1094,1102,1103,1105,1106,1116,1117,1123,1128,1134,1151,1161,1174,1177,1178,1190,1192,1195,1200,1210,1211,1216,1222,1226,1229,1230,1231,1241,1261,1263,1268,1269,1270,1272,1273,1298,1299,1307,1316,1321,1322,1330,1336,1345,1351,1359,1360,1361,1362,1364,1374,1377,1385,1392,1393,1407,1411,1412,1421,1429,1437,1442,1475,1488,1491,1497,1502,1509,1512,1533,1536,1547,1551,1553,1556,1557,1559,1562,1566,1567,1569,1577,1578,1582,1583,1586,1600,1601,1607,1614,1618,1624,1637,1641,1659,1665,1670,1691,1694,1706,1710,1725,1732,1736,1740,1741,1745,1746,1748,1752,1760,1762,1763,1770,1776,1780,1788,1793,1798,1800,1801,1804,1813,1814,1817,1818,1820,1830,1831,1839,1840,1842,1861,1868,1876,1877,1882,1885,1898,1904,1921,1929,1939,1941,1949,1963,1965,1966,1967,1991,1992,2001,2012,2024,2025,2029,2030,2045,2066,2069,2083,2089,2093,2095,2096,2097,2111,2112,2113,2114,2123,2124,2132,2144,2151,2153,2155,2163,2164,2166,2167,2174,2175,2176,2186,2190,2206,2213,2219,2225,2230,2236,2243,2254,2256,2268,2272,2284,2290,2291,2297,2303,2307,2316,2323,2330,2332,2342,2358,2366,2372,2377,2392,2404,2414,2417,2418,2425,2441,2442,2453,2458,2472,2473,2475,2477,2482,2497,2501,2505,2523,2525,2528,2529,2531,2538,2542,2546,2564,2567,2576,2579,2592,2594,2601,2603,2618,2619,2627,2631,2636,2644,2659,2663,2667,2668,2677,2678,2681,2689,2700,2703,2705,2706,2709,2718,2724,2737,2738,2743,2744,2748,2752,2758,2759,2776,2779,2786,2825,2846,2847,2851,2862,2873,2881,2883,2887,2890,2895,2897,2901,2909,2916,2927,2929,2930,2934,2957,2958,2967,2968,2986,2988,2991,2994,2996,3001,3013,3014,3017,3023,3028,3031,3034,3036,3037,3041,3050,3054,3065,3066,3074,3078,3079,3080,3084,3088,3093,3101,3116,3126,3130,3132,3133,3156,3157,3167,3168,3171,3177,3181,3198,3199,3221,3228,3230,3231,3233,3239,3242,3254,3258,3259,3263,3279,3296,3308,3319,3323,3324,3325,3343,3345,3346,3351,3352,3371,3372,3373,3388,3393,3401,3407,3408,3410,3417,3420,3439,3441,3465,3466,3467,3469,3472,3476,3496,3501,3506,3512,3521,3523,3530,3534,3535,3537,3538,3544,3545,3553,3561,3565,3576,3586,3591,3599,3604,3616,3622,3625,3635,3640,3641,3646,3651,3663,3672,3676,3681,3691,3696,3704,3705,3706,3707,3710,3712,3722,3723,3728,3730,3740,3742,3743,3746,3747,3755,3763,3766,3767,3774,3781,3791,3794,3799,3800,3808,3811,3813,3814,3823,3825,3836,3840,3846,3878,3879,3889,3893,3895,3907,3920,3940,3947,3953,3955,3965,3966,3967,3968,3969,3988,3994,4005,4008,4009,4012,4022,4027,4028,4032,4048,4059,4065,4069,4073,4074,4077,4078,4079,4080,4086,4097,4101,4108,4119,4128,4138,4165,4176,4182,4183,4186,4190,4192,4195,4199,4212,4215,4219,4225,4242,4248,4266,4269,4272,4277,4280,4282,4291,4295,4310,4311,4320,4321,4324,4328,4335,4339,4343,4353,4354,4371,4373,4388,4393,4397,4404,4407,4419,4420,4425,4428,4433,4441,4451,4477,4480,4483,4487,4493,4495,4496,4498,4505,4521,4533,4535,4540,4544,4559,4564,4566,4570,4575,4580,4602,4604,4613,4623,4630,4647,4652,4663,4668,4669,4671,4679,4684,4695,4698,4715,4720,4728,4732,4733,4735,4737,4741,4747,4748,4755,4775,4778,4782,4783,4785,4786,4788,4794,4816,4817,4820,4821,4825,4832,4862,4865,4876,4882,4885,4886,4893,4895,4905,4918,4919,4922,4926,4934,4939,4940,4945,4954,4958,4961,4962,4971,4973,4974,4978,4985,5009,5013,5025,5028,5030,5031,5035,5040,5058,5060,5063,5083,5089,5095,5107,5108,5110,5112,5113,5115,5148,5149,5155,5159,5166,5171,5179,5183,5203,5209,5210,5225,5226,5232,5240,5246,5247,5249,5253,5264,5268,5270,5285,5287,5294,5298,5303,5306,5308,5312,5314,5326,5329,5332,5333,5340,5349,5350,5356,5360,5366,5374,5385,5388,5389,5393,5398,5399,5404,5407,5410,5413,5417,5420,5425,5440,5444,5453,5460,5461,5462,5464,5470,5474,5476,5481,5484,5486,5491,5494,5506,5507,5509,5510,5511,5525,5548,5557,5560,5581,5584,5585,5599,5609,5630,5634,5637,5645,5656,5679,5680,5688,5691,5694,5696,5706,5710,5715,5722,5729,5737,5742,5745,5746,5748,5754,5756,5759,5767,5769,5770,5775,5776,5780,5784,5791,5802,5806,5812,5815,5832,5835,5837,5839,5845,5856,5858,5865,5866,5868,5873,5877,5878,5893,5895,5899,5909,5913,5925,5940,5945,5948,5951,5956,5957,5958,5965,5966,5971,5973,5974,5984,5991,5992,5996,6002,6007,6010,6022,6029,6032,6042,6043,6047,6053,6071,6076,6080,6088,6092,6093,6095,6097,6106,6107,6112,6119,6123,6131,6135,6137,6141,6144,6159,6163,6169,6202,6222,6238,6248,6256,6262,6265,6267,6286,6297,6299,6302,6305,6307,6311,6315,6318,6321,6330,6336,6341,6345,6347,6359,6360,6369,6374,6384,6410,6416,6424,6430,6447,6450,6456,6457,6464,6481,6483,6484,6488,6499,6504,6509,6518,6521,6525,6552,6563,6567,6568,6570,6577,6581,6582,6584,6586,6590,6592,6597,6606,6609,6620,6623,6624,6635,6637,6643,6644,6646,6651,6658,6660,6668,6673,6685,6695,6705,6708,6713,6714,6720,6732,6733,6735,6747,6761,6763,6766,6768,6778,6782,6783,6788,6795,6799,6814,6817,6823,6828,6829,6840,6846,6850,6851,6856,6864,6866,6871,6874,6876,6879,6883,6902,6908,6923,6932,6934,6936,6940,6948,6949,6956,6961,6967,6975,6980,6999,7002,7006,7011,7014,7020,7021,7023,7043,7048,7064,7068,7072,7076,7078,7083,7085,7088,7095,7100,7109,7120,7123,7127,7136,7144,7149,7153,7175,7179,7186,7195,7199,7205,7209,7225,7231,7241,7246,7249,7256,7265,7267,7286,7287,7293,7297,7300,7303,7305,7333,7361,7363,7365,7367,7372,7376,7377,7382,7386,7393,7406,7414,7418,7422,7430,7440,7441,7447,7450,7458,7460,7470,7471,7472,7482,7483,7496,7497,7499,7508,7512,7517,7524,7528,7529,7538,7548,7558,7587,7591,7592,7596,7610,7614,7616,7623,7631,7637,7642,7653,7660,7661,7666,7668,7670,7685,7688,7689,7690,7691,7708,7712,7717,7725,7726,7731,7739,7742,7752,7758,7762,7771,7773,7778,7788,7789,7792,7794,7802,7803,7804,7818,7826,7833,7835,7836,7848,7851,7862,7877,7880,7893,7896,7903,7915,7919,7921,7922,7924,7939,7947,7948,7957,7965,7967,7974,7975,7976,7977,7982,7987,7998,8000,8002,8005,8008,8009,8020,8030,8032,8033,8034,8047,8059,8066,8074,8083,8085,8087,8089,8091,8109,8114,8119,8124,8133,8145,8150,8157,8170,8176,8180,8190,8193,8195,8200,8203,8209,8213,8215,8217,8238,8243,8245,8248,8250,8251,8254,8257,8259,8270,8277,8278,8282,8288,8291,8307,8308,8309,8315,8319,8322,8324,8327,8328,8335,8342,8350,8351,8359,8363,8370,8375,8386,8389,8393,8396,8399,8411,8413,8417,8418,8434,8438,8441,8444,8445,8448,8454,8459,8465,8468,8471,8475,8482,8484,8501,8504,8506,8509,8520,8521,8535,8536,8546,8552,8556,8557,8562,8564,8570,8573,8577,8603,8608,8616,8620,8628,8629,8633,8634,8637,8666,8679,8681,8682,8699,8704,8715,8720,8722,8726,8746,8747,8755,8757,8765,8767,8777,8780,8790,8791,8801,8802,8817,8821,8827,8829,8838,8850,8857,8864,8871,8873,8874,8876,8886,8889,8893,8901,8903,8912,8921,8930,8931,8936,8938,8943,8944,8950,8956,8962,8963,8974,8975,8977,8978,8981,8988,8994,8997,9001,9014,9015,9019,9026,9033,9044,9047,9049,9052,9053,9057,9064,9077,9078,9082,9097,9098,9099,9101,9102,9107,9110,9112,9115,9118,9119,9129,9136,9139,9149,9151,9152,9155,9156,9158,9160,9163,9170,9176,9177,9180,9181,9182,9184,9192,9198,9200,9205,9208,9214,9218,9219,9221,9223,9226,9230,9237,9245,9250,9251,9260,9281,9282,9300,9303,9304,9314,9320,9321,9327,9334,9340,9342,9346,9351,9357,9359,9362,9365,9368,9371,9377,9379,9385,9391,9398,9403,9404,9405,9417,9418,9423,9429,9447,9452,9457,9459,9461,9468,9476,9479,9487,9499,9508,9510,9519,9522,9523,9531,9533,9534,9536,9547,9553,9558,9561,9562,9564,9566,9569,9571,9592,9602,9607,9610,9611,9612,9613,9614,9617,9624,9628,9631,9634,9635,9641,9644,9647,9653,9659,9673,9681,9691,9700,9706,9723,9732,9741,9743,9744,9748,9749,9750,9751,9754,9757,9759,9764,9772,9778,9779,9787,9800,9805,9807,9820,9822,9823,9828,9830,9846,9850,9854,9863,9871,9875,9876,9880,9893,9920,9931,9940,9943,9944,9950,9951,9953,9954,9959,9970,9977,9989,9994,9999,10002,10003,10005,10006,10013,10016,10017,10018,10021,10024,10028,10029,10055,10057,10059,10063,10068,10072,10078,10089,10114,10116,10117,10121,10128,10139,10141,10148,10152,10160,10163,10166,10172,10173,10178,10198,10200,10203,10205,10218,10221,10223,10229,10236,10243,10254,10255,10266,10267,10274,10275,10282,10286,10287,10301,10313,10319,10321,10323,10324,10325,10326,10329,10331,10334,10337,10338,10340,10349,10350,10367,10370,10376,10380,10384,10390,10392,10393,10396,10401,10402,10404,10406,10407,10410,10412,10413,10415,10416,10418,10421,10427,10430,10431,10433,10436,10447,10460,10466,10474,10476,10482,10485,10489,10491,10493,10496,10500,10501,10502,10504,10511,10512,10513,10521,10522,10523,10525,10530,10541,10542,10544,10546,10553,10556,10557,10560,10568,10570,10572,10573,10579,10580,10615,10619,10627,10630,10639,10642,10643,10647,10650,10651,10654,10655,10656,10669,10670,10672,10679,10682,10698,10701,10709,10711,10724,10732,10735,10737,10740,10751,10764,10779,10781,10782,10784,10798,10807,10808,10810,10812,10820,10827,10829,10844,10846,10864,10871,10880,10884,10887,10888,10896,10898,10899,10900,10902,10916,10922,10925,10929,10935,10938,10943,10945,10961,10963,10968,10970,10974,10978,10983,10984,10987,10989,10990,11000,11004,11005,11007,11009,11017,11019,11020,11021,11028,11029,11036,11040,11041,11052,11055,11058,11065,11074,11076,11081,11084,11085,11094,11099,11102,11104,11106,11116,11120,11129,11130,11133,11137,11147,11166,11172,11174,11177,11178,11187,11195,11203,11207,11218,11219,11220,11237,11238,11241,11244,11246,11248,11262,11265,11279,11285,11306,11309,11318,11319,11320,11322,11326,11339,11341,11342,11345,11346,11349,11353,11357,11360,11361,11370,11376,11377,11392,11393,11395,11397,11410,11413,11414,11415,11421,11424,11429,11447,11458,11471,11473,11481,11482,11483,11488,11492,11495,11498,11504,11517,11518,11519,11521,11522,11527,11532,11537,11540,11542,11547,11553,11557,11569,11571,11584,11587,11588,11591,11593,11599,11622,11626,11628,11634,11636,11638,11644,11647,11652,11666,11668,11669,11676,11678,11682,11683,11688,11689,11690,11691,11696,11697,11702,11707,11710,11711,11713,11715,11718,11722,11724,11725,11729,11742,11750,11757,11759,11761,11765,11767,11773,11782,11786,11790,11791,11796,11798,11801,11804,11810,11812,11814,11817,11822,11827,11837,11838,11844,11845,11846,11851,11857,11863,11866,11873,11880,11883,11887,11893,11905,11920,11922,11923,11928,11932,11936,11938,11944,11945,11948,11949,11954,11956,11967,11968,11976,11985,11987,11998,11999,12009,12015,12018,12023,12042,12066,12068,12069,12071,12075,12076,12096,12103,12108,12109,12110,12116,12130,12134,12136,12155,12169,12171,12177,12180,12181,12184,12188,12191,12193,12207,12211,12213,12215,12219,12225,12231,12232,12233,12242,12250,12255,12260,12281,12288,12296,12297,12307,12316,12322,12335,12336,12338,12344,12351,12352,12353,12371,12379,12380,12382,12384,12385,12389,12390,12393,12401,12402,12404,12411,12413,12423,12441,12450,12453,12456,12461,12463,12464,12465,12469,12473,12477,12494,12495,12499,12505,12515,12519,12520,12521,12525,12532,12539,12553,12554,12559,12570,12572,12578,12579,12582,12586,12603,12605,12607,12608,12615,12617,12620,12635,12636,12639,12645,12646,12654,12659,12662,12667,12669,12674,12676,12681,12687,12692,12705,12712,12713,12714,12715,12718,12719,12720,12722,12725,12730,12732,12743,12754,12762,12768,12770,12776,12785,12788,12793,12799,12806,12813,12818,12819,12821,12822,12823,12830,12831,12835,12839,12844,12846,12858,12859,12878,12879,12884,12886,12887,12889,12890,12893,12899,12910,12916,12918,12919,12929,12933,12937,12940,12945,12948,12953,12956,12961,12974,12975,12979,12980,12984,12995,13005,13014,13020,13021,13023,13024,13033,13034,13043,13049,13053,13060,13081,13082,13087,13091,13115,13133,13148,13151,13159,13162,13166,13180,13181,13186,13190,13192,13193,13198,13204,13211,13212,13219,13232,13236,13241,13242,13245,13247,13257,13264,13266,13267,13270,13272,13280,13281,13294,13296,13298,13299,13302,13309,13312,13313,13315,13321,13323,13326,13327,13328,13333,13341,13343,13345,13347,13354,13356,13361,13363,13365,13376,13381,13384,13387,13388,13390,13395,13409,13411,13416,13420,13438,13439,13444,13451,13466,13467,13469,13470,13471,13475,13484,13490,13491,13495,13501,13502,13507,13513,13516,13523,13526,13535,13538,13541,13542,13549,13551,13557,13558,13559,13565,13566,13569,13577,13579,13583,13588,13594,13597,13608,13609,13612,13614,13618,13625,13628,13629,13638,13642,13643,13646,13647,13648,13655,13660,13667,13674,13685,13687,13688,13693,13699,13708,13718,13723,13732,13733,13734,13740,13741,13763,13764,13774,13780,13781,13785,13791,13795,13804,13830,13834,13835,13842,13854,13857,13859,13869,13874,13875,13877,13880,13888,13890,13896,13900,13903,13905,13911,13912,13915,13926,13930,13935,13945,13947,13953,13960,13962,13968,13970,13980,13983,13986,13995,13999,14001,14002,14003,14006,14013,14018,14019,14024,14025,14028,14047,14051,14053,14055,14060,14062,14069,14073,14079,14087,14088,14093,14097,14099,14101,14102,14106,14107,14114,14115,14120,14123,14125,14126,14130,14133,14137,14138,14149,14158,14160,14161,14162,14163,14164,14170,14172,14173,14180,14181,14186,14187,14189,14195,14196,14204,14205,14207,14208,14221,14224,14225,14227,14228,14234,14236,14237,14247,14250,14254,14256,14262,14265,14268,14272,14273,14277,14281,14283,14284,14291,14315,14318,14321,14322,14323,14324,14325,14326,14327,14346,14349,14354,14357,14361,14362,14363,14375,14376,14378,14380,14383,14387,14392,14393,14396,14400,14401,14406,14409,14410,14413,14416,14417,14424,14426,14432,14433,14443,14446,14447,14455,14456,14457,14459,14460,14468,14470,14471,14481,14485,14488,14490,14496,14500,14501,14504,14520,14522,14523,14528,14535,14537,14547,14550,14555,14557,14559,14562,14577,14581,14583,14584,14586,14588,14592,14594,14596,14599,14600,14606,14615,14623,14627,14628,14630,14631,14632,14634,14635,14636,14639,14640,14641,14642,14644,14647,14650,14657,14664,14667,14671,14677,14692,14705,14706,14708,14711,14713,14724,14726,14730,14734,14736,14739,14740,14742,14745,14758,14759,14762,14764,14771,14783,14784,14785,14790,14805,14808,14812,14814,14816,14820,14824,14825,14826,14832,14840,14847,14851,14857,14861,14862,14868,14872,14873,14876,14880,14882,14883,14887,14891,14892,14896,14900,14901,14902,14906,14920,14928,14935,14939,14940,14945,14946,14949,14950,14955,14960,14964,14971,14972,14980,14987,14988,14990,14992,14997,15002,15008,15014,15017,15019,15020,15027,15029,15031,15035,15038,15050,15052,15064,15069,15074,15076,15077,15080,15081,15084,15089,15099,15109,15110,15111,15117,15118,15124,15130,15133,15139,15143,15144,15146,15149,15151,15153,15154,15162,15164,15179,15181,15182,15186,15194,15196,15198,15203,15206,15207,15214,15215,15217,15218,15222,15228,15232,15233,15250,15254,15257,15258,15265,15276,15284,15287,15298,15302,15303,15306,15315,15316,15318,15326,15338,15340,15341,15343,15347,15349,15351,15358,15360,15372,15377,15391,15393,15398,15401,15403,15407,15408,15410,15412,15415,15419,15420,15428,15429,15441,15449,15454,15456,15460,15469,15470,15476,15479,15480,15487,15489,15490,15491,15502,15503,15506,15508,15509,15525,15527,15528,15535,15536,15537,15539,15549,15550,15551,15553,15562,15568,15570,15574,15576,15578,15580,15582,15593,15595,15599,15602,15610,15612,15615,15618,15623,15627,15631,15639,15644,15646,15647,15654,15662,15663,15664,15665,15669,15675,15681,15682,15695,15705,15711,15716,15720,15723,15728,15730,15733,15736,15738,15739,15742,15749,15753,15756,15758,15762,15767,15769,15772,15773,15784,15787,15791,15793,15797,15800,15805,15808,15809,15812,15815,15816,15817,15818,15822,15826,15830,15833,15834,15836,15837,15840,15841,15846,15847,15856,15859,15862,15864,15870,15871,15884,15893,15896,15900,15901,15904,15907,15909,15913,15919,15922,15924,15929,15933,15934,15938,15948,15953,15957,15958,15964,15966,15970,15976,15980,15981,15983,16004,16005,16007,16010,16017,16024,16035,16036,16039,16042,16044,16045,16057,16079,16080,16090,16093,16095,16100,16103,16116,16117,16131,16143,16144,16153,16159,16186,16192,16195,16199,16210,16212,16213,16221,16236,16251,16252,16257,16261,16265,16269,16273,16278,16280,16283,16288,16290,16294,16295,16301,16306,16311,16313,16315,16321,16322,16325,16326,16331,16332,16334,16336,16337,16338,16344,16350,16351,16353,16362,16371,16373,16390,16409,16412,16416,16417,16419,16420,16423,16424,16428,16429,16432,16441,16444,16455,16461,16463,16464,16467,16469,16471,16478,16488,16499,16513,16516,16519,16524,16530,16531,16534,16538,16540,16544,16546,16548,16564,16568,16577,16589,16593,16596,16598,16599,16600,16602,16604,16610,16613,16638,16642,16643,16646,16647,16661,16670,16673,16676,16678,16680,16683,16685,16698,16699,16703,16705,16709,16715,16719,16733,16736,16742,16746,16755,16773,16778,16790,16791,16794,16802,16808,16809,16828,16830,16835,16837,16838,16840,16852,16860,16866,16867,16880,16881,16891,16892,16897,16908,16912,16921,16922,16928,16929,16940,16969,16970,16973,16975,16980,16981,16994,17009,17014,17031,17032,17034,17035,17047,17051,17052,17053,17057,17058,17068,17077,17079,17082,17094,17098,17100,17103,17110,17112,17113,17114,17117,17118,17120,17122,17124,17125,17133,17136,17141,17144,17145,17160,17167,17172,17180,17185,17187,17189,17209,17216,17223,17233,17237,17246,17249,17253,17256,17259,17260,17263,17268,17274,17279,17286,17289,17290,17291,17304,17305,17306,17309,17324,17328,17334,17337,17340,17341,17347,17349,17354,17363,17367,17370,17377,17378,17381,17383,17384,17390,17407,17408,17417,17419,17423,17429,17434,17439,17443,17445,17453,17471,17478,17482,17484,17485,17488,17489,17491,17495,17502,17503,17509,17512,17518,17526,17527,17534,17555,17559,17562,17566,17569,17573,17579,17580,17582,17585,17586,17587,17592,17599,17604,17608,17609,17610,17626,17627,17629,17632,17633,17634,17642,17643,17646,17650,17651,17676,17677,17680,17683,17684,17695,17699,17701,17705,17712,17723,17730,17731,17736,17738,17751,17752,17758,17761,17764,17765,17773,17777,17779,17783,17785,17790,17794,17803,17806,17821,17823,17826,17827,17831,17843,17845,17848,17850,17860,17861,17874,17883,17888,17892,17904,17905,17921,17931,17939,17940,17941,17942,17945,17960,17965,17976,17980,17983,17988,17992,17994,17999,18000,18001,18006,18011,18013,18014,18018,18030,18031,18039,18040,18041,18049,18056,18057,18058,18071,18077,18078,18085,18092,18094,18102,18103,18104,18109,18119,18123,18125,18132,18135,18136,18142,18146,18153,18161,18164,18168,18178,18179,18181,18187,18191,18193,18201,18203,18204,18205,18206,18207,18211,18218,18220,18225,18226,18229,18242,18246,18248,18252,18253,18256,18265,18272,18282,18291,18292,18295,18296,18297,18299,18314,18332,18334,18338,18340,18341,18346,18352,18354,18355,18358,18359,18366,18368,18370,18386,18389,18390,18391,18392,18394,18399,18425,18434,18440,18441,18445,18447,18448,18451,18452,18453,18456,18457,18464,18469,18473,18474,18478,18480,18486,18497,18498,18505,18508,18514,18519,18521,18522,18529,18536,18537,18538,18544,18556,18558,18580,18583,18589,18591,18593,18599,18600,18604,18628,18629,18630,18637,18644,18646,18647,18650,18653,18658,18664,18670,18671,18674,18682,18690,18693,18705,18708,18717,18729,18734,18735,18746,18749,18750,18759,18768,18778,18779,18782,18783,18787,18788,18797,18804,18812,18817,18829,18830,18837,18842,18844,18848,18849,18852,18856,18861,18862,18869,18871,18877,18887,18906,18908,18923,18924,18936,18943,18955,18958,18969,18991,18992,18996,18997,19001,19002,19003,19012,19018,19022,19025,19043,19053,19054,19062,19063,19070,19074,19085,19095,19102,19106,19107,19116,19119,19120,19124,19126,19128,19130,19140,19158,19165,19176,19178,19180,19181,19185,19192,19198,19214,19218,19223,19225,19231,19234,19248,19255,19258,19279,19287,19288,19289,19300,19303,19304,19311,19312,19323,19324,19333,19340,19343,19348,19356,19372,19380,19381,19383,19386,19388,19390,19392,19393,19401,19404,19414,19417,19419,19420,19454,19458,19461,19466,19468,19489,19497,19498,19500,19501,19510,19519,19523,19530,19536,19549,19563,19564,19565,19576,19587,19589,19595,19607,19611,19619,19632,19636,19650,19667,19670,19671,19682,19689,19690,19691,19692,19697,19701,19704,19711,19713,19724,19726,19727,19730,19731,19746,19751,19754,19759,19766,19769,19776,19781,19785,19788,19797,19800,19808,19823,19833,19835,19842,19843,19849,19856,19863,19864,19867,19870,19872,19873,19876,19882,19893,19904,19912,19925,19935,19944,19945,19950,19951,19953,19959,19962,19963,19973,19975,19976,19989,20005,20010,20025,20028,20036,20038,20040,20043,20051,20052,20055,20061,20074,20080,20098,20107,20130,20131,20135,20139,20140,20152,20156,20157,20158,20159,20166,20174,20176,20190,20195,20197,20210,20219,20221,20229,20240,20241,20245,20256,20257,20264,20265,20267,20280,20283,20286,20291,20293,20299,20310,20331,20340,20347,20367,20371,20375,20379,20391,20395,20401,20406,20414,20425,20426,20428,20430,20433,20440,20444,20446,20447,20457,20473,20475,20485,20486,20488,20491,20498,20499,20503,20506,20514,20520,20530,20536,20537,20546,20562,20571,20585,20588,20590,20593,20599,20601,20604,20608,20609,20612,20613,20614,20615,20616,20620,20627,20638,20641,20645,20651,20653,20658,20662,20669,20703,20705,20706,20710,20714,20728,20730,20739,20741,20743,20747,20750,20753,20755,20764,20765,20780,20790,20791,20799,20807,20809,20829,20832,20839,20845,20879,20882,20887,20896,20897,20899,20903,20904,20915,20923,20929,20936,20943,20944,20948,20955,20957,20960,20965,20972,20975,20978,20988,21001,21018,21028,21042,21055,21062,21063,21065,21066,21070,21071,21088,21098,21104,21109,21114,21124,21129,21137,21145,21147,21148,21149,21151,21157,21161,21164,21174,21182,21185,21204,21206,21209,21210,21211,21222,21236,21242,21245,21248,21254,21255,21286,21289,21292,21312,21315,21316,21317,21326,21329,21337,21349,21351,21355,21370,21379,21380,21385,21390,21395,21403,21411,21415,21416,21419,21420,21425,21426,21427,21430,21439,21443,21444,21450,21454,21467,21476,21478,21479,21498,21500,21501,21506,21507,21512,21519,21536,21547,21552,21555,21563,21564,21566,21567,21575,21578,21582,21585,21590,21598,21604,21627,21640,21641,21645,21647,21664,21671,21673,21674,21677,21683,21686,21688,21691,21692,21697,21699,21704,21705,21707,21712,21721,21723,21736,21737,21746,21748,21750,21752,21762,21763,21764,21767,21774,21777,21780,21785,21787,21788,21793,21808,21810,21814,21824,21825,21837,21842,21847,21862,21866,21872,21878,21879,21881,21882,21884,21897,21901,21902,21903,21911,21912,21920,21923,21928,21938,21939,21945,21948,21960,21963,21970,21972,21976,21977,21978,21983,21987,21993,21995,21996,22003,22004,22005,22013,22015,22016,22017,22019,22020,22021,22026,22027,22036,22040,22043,22077,22089,22090,22092,22094,22100,22101,22103,22105,22126,22140,22149,22152,22157,22163,22166,22180,22181,22185,22186,22187,22189,22192,22204,22207,22215,22221,22223,22224,22233,22246,22248,22262,22265,22277,22279,22284,22288,22290,22292,22298,22309,22310,22313,22314,22320,22323,22324,22327,22329,22331,22350,22353,22359,22367,22373,22376,22378,22379,22386,22388,22389,22400,22404,22412,22414,22419,22422,22425,22428,22443,22444,22449,22450,22457,22459,22461,22467,22478,22479,22501,22517,22518,22526,22530,22531,22545,22551,22552,22556,22560,22562,22581,22582,22585,22587,22594,22602,22604,22607,22616,22617,22620,22626,22630,22638,22639,22641,22643,22648,22658,22660,22662,22668,22670,22672,22679,22683,22685,22687,22689,22690,22702,22703,22708,22709,22712,22721,22730,22741,22744,22746,22755,22759,22764,22768,22774,22777,22778,22781,22783,22786,22799,22803,22805,22817,22819,22820,22831,22832,22835,22836,22843,22844,22854,22861,22862,22864,22867,22868,22882,22884,22890,22895,22898,22905,22909,22914,22917,22922,22926,22927,22930,22934,22948,22952,22961,22962,22965,22968,22995,23011,23014,23019,23031,23032,23038,23043,23046,23047,23054,23057,23058,23059,23061,23062,23070,23083,23087,23098,23105,23107,23110,23111,23114,23125,23126,23128,23130,23133,23135,23137,23142,23143,23144,23150,23152,23153,23155,23159,23173,23178,23179,23182,23185,23188,23209,23214,23234,23236,23249,23257,23262,23268,23274,23279,23280,23286,23287,23289,23292,23300,23302,23307,23310,23314,23315,23319,23327,23334,23344,23345,23346,23348,23356,23358,23361,23367,23368,23369,23370,23373,23376,23379,23384,23385,23390,23398,23404,23405,23406,23409,23413,23423,23431,23449,23454,23474,23478,23479,23480,23483,23485,23494,23495,23496,23498,23511,23513,23516,23518,23524,23537,23549,23557,23567,23570,23572,23573,23579,23594,23597,23612,23623,23624,23627,23632,23639,23645,23647,23651,23652,23656,23657,23672,23673,23676,23677,23682,23690,23691,23697,23699,23706,23712,23720,23726,23728,23748,23751,23757,23758,23759,23764,23766,23768,23782,23784,23790,23794,23797,23798,23799,23802,23817,23833,23834,23836,23840,23841,23847,23848,23856,23857,23862,23873,23885,23887,23889,23895,23902,23906,23911,23912,23913,23915,23917,23919,23921,23926,23927,23928,23935,23936,23938,23948,23950,23954,23959,23960,23968,23975,23998,24006,24013,24023,24050,24053,24054,24060,24061,24064,24068,24069,24092,24099,24100,24113,24117,24121,24134,24136,24137,24138,24139,24143,24149,24155,24162,24163,24166,24171,24176,24179,24184,24187,24191,24192,24196,24209,24214,24217,24225,24234,24244,24248,24250,24251,24267,24270,24271,24272,24273,24276,24288,24298,24306,24310,24312,24319,24322,24324,24325,24329,24339,24351,24369,24376,24387,24389,24393,24399,24400,24404,24418,24428,24438,24444,24457,24459,24467,24470,24471,24483,24494,24505,24508,24513,24516,24540,24542,24548,24549,24552,24553,24559,24563,24565,24566,24570,24573,24576,24579,24583,24588,24598,24608,24615,24621,24626,24628,24630,24631,24635,24645,24659,24665,24667,24668,24672,24685,24688,24690,24695,24723,24724,24727,24728,24733,24735,24746,24751,24762,24764,24765,24766,24771,24776,24780,24783,24794,24801,24810,24812,24825,24829,24831,24836,24841,24846,24851,24858,24860,24864,24866,24868,24874,24878,24886,24889,24893,24894,24900,24902,24905,24910,24914,24925,24938,24939,24955,24977,24982,24999,25006,25009,25012,25021,25025,25030,25037,25038,25043,25047,25048,25049,25064,25071,25079,25088,25089,25092,25094,25099,25119,25124,25125,25136,25148,25152,25155,25157,25159,25174,25181,25190,25191,25200,25208,25229,25233,25238,25244,25249,25251,25256,25269,25270,25274,25280,25285,25288,25296,25298,25301,25302,25321,25326,25327,25329,25331,25335,25351,25364,25367,25377,25380,25407,25410,25431,25432,25441,25444,25446,25450,25454,25456,25470,25478,25484,25485,25487,25505,25511,25513,25527,25533,25535,25537,25540,25541,25543,25553,25557,25560,25561,25572,25590,25593,25601,25602,25606,25607,25609,25620,25625,25626,25630,25634,25636,25650,25667,25668,25670,25671,25687,25692,25693,25694,25695,25710,25711,25714,25717,25718,25720,25722,25724,25725,25727,25733,25736,25739,25742,25743,25744,25758,25760,25761,25765,25772,25807,25820,25825,25830,25862,25873,25893,25896,25905,25908,25921,25923,25927,25953,25956,25960,25965,25972,25974,25989,25992,25995,25999,26002,26006,26013,26014,26016,26030,26032,26033,26043,26056,26064,26070,26073,26076,26086,26087,26095,26102,26110,26121,26125,26127,26134,26137,26141,26164,26173,26175,26179,26189,26191,26200,26205,26216,26218,26219,26234,26239,26245,26248,26251,26259,26273,26275,26282,26289,26306,26309,26311,26319,26323,26330,26331,26339,26343,26344,26347,26348,26352,26355,26356,26358,26360,26361,26371,26375,26383,26395,26403,26404,26408,26411,26420,26424,26434,26438,26440,26441,26447,26457,26459,26469,26485,26487,26492,26493,26495,26498,26500,26501,26520,26532,26537,26549,26553,26554,26555,26559,26560,26561,26570,26572,26574,26583,26587,26618,26631,26638,26644,26648,26651,26663,26673,26679,26690,26691,26695,26715,26718,26730,26732,26735,26737,26739,26740,26744,26745,26746,26753,26760,26766,26772,26775,26782,26790,26791,26801,26802,26806,26810,26812,26825,26834,26838,26846,26856,26860,26864,26865,26874,26877,26890,26891,26901,26908,26916,26921,26922,26925,26929,26936,26938,26941,26950,26972,26977,26981,26984,26997,]

    @staticmethod
    def num_train_examples(): return 22000

    @staticmethod
    def num_test_examples(): return 5000*4  # 5000 images split into quadrants

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_data(train):
        dataset = torchvision.datasets.EuroSAT(root=os.path.join(get_platform().dataset_root, 'eurosat'), download=True)
        # default train/test split
        train_mask = np.ones(len(dataset), dtype=bool)
        train_mask[Dataset.TEST_SPLIT] = False
        mask = train_mask if train else np.logical_not(train_mask)
        data, targets = [], []
        test_transform = torchvision.transforms.FiveCrop(32)
        for include_example, (example, label) in zip(mask, dataset):
            if include_example:
                if train:
                    data.append(example)
                    targets.append(label)
                else:  # split 64x64 test images into four 32x32 images
                    for quadrant in test_transform(example)[:4]:
                        data.append(quadrant)
                        targets.append(label)
        data = np.stack(data, axis=0)
        targets = np.stack(targets, axis=0)
        return data, targets

    @staticmethod
    def get_train_set(use_augmentation, train_split=None):
        augment = [
            # get all eight 90 degree rotation + flip symmetries including identity
            torchvision.transforms.RandomChoice([
                lambda x: torchvision.transforms.functional.rotate(x, 0),
                lambda x: torchvision.transforms.functional.hflip(x),
                ]),
            torchvision.transforms.RandomChoice([
                lambda x: torchvision.transforms.functional.rotate(x, 0),
                lambda x: torchvision.transforms.functional.rotate(x, 90),
                lambda x: torchvision.transforms.functional.rotate(x, 180),
                lambda x: torchvision.transforms.functional.rotate(x, 270),
            ]),
            torchvision.transforms.RandomCrop((32, 32))
        ]
        data, targets = Dataset.get_data_split(True, train_split)
        return Dataset(data, targets, augment if use_augmentation else  \
                       [torchvision.transforms.CenterCrop((32, 32))])

    @staticmethod
    def get_test_set(test_split=None):
        data, targets = Dataset.get_data_split(False, test_split)
        return Dataset(data, targets)

    def __init__(self,  examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize(
                                        mean=[0.344, 0.380, 0.408], std=[0.203, 0.137, 0.116])])

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
