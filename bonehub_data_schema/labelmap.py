from enum import Enum, unique


@unique
class BoneLabelMap(Enum):
    """Unified bone labels."""

    BACKGROUND = 0  # background label

    ## axial skeleton
    # skull
    SKULL = 1000  # whole skull including cranium and facial bones
    SKULL_CRANIAL = 1100  # whole cranial only
    SKULL_FACIAL = 1200  # whole facial bones only
    SKULL_MANDIBLE = 1201  # mandible bone which is part of facial bones
    SKULL_MAXILLA = 1202  # maxilla bone which is part of facial bones

    # spinal column
    VERTEBRAE = 2000  # whole vertebral column without sacrum and coccyx

    VERTEBRAE_CERVICAL = 2100  # whole cervical vertebrae from C1 to C7
    VERTEBRA_C1 = 2101
    VERTEBRA_C2 = 2102
    VERTEBRA_C3 = 2103
    VERTEBRA_C4 = 2104
    VERTEBRA_C5 = 2105
    VERTEBRA_C6 = 2106
    VERTEBRA_C7 = 2107
    VERTEBRAE_THORACIC = 2200  # whole thoracic vertebrae from T1 to T12
    VERTEBRA_T1 = 2201
    VERTEBRA_T2 = 2202
    VERTEBRA_T3 = 2203
    VERTEBRA_T4 = 2204
    VERTEBRA_T5 = 2205
    VERTEBRA_T6 = 2206
    VERTEBRA_T7 = 2207
    VERTEBRA_T8 = 2208
    VERTEBRA_T9 = 2209
    VERTEBRA_T10 = 2210
    VERTEBRA_T11 = 2211
    VERTEBRA_T12 = 2212
    VERTEBRAE_LUMBAR = 2300  # whole lumbar vertebrae from L1 to L5
    VERTEBRA_L1 = 2301
    VERTEBRA_L2 = 2302
    VERTEBRA_L3 = 2303
    VERTEBRA_L4 = 2304
    VERTEBRA_L5 = 2305
    VERTEBRA_L6 = 2306

    SACRUM = 2400  # whole sacrum
    SACRUM_WITHOUT_S1 = 2401  # whole sacrum excluding S1 vertebra
    VERTEBRA_S1 = 2402  # only S1 vertebra of sacrum

    # thorax
    RIBS = 3000  # whole rib cage excluding sternum
    RIBS_WITH_STERNUM = 3001  # whole rib cage including sternum
    # The digit after the underscore is the rib number (1-12 from top to bottom).
    RIB_1_LEFT = 3101
    RIB_1_RIGHT = 3102
    RIB_2_LEFT = 3103
    RIB_2_RIGHT = 3104
    RIB_3_LEFT = 3105
    RIB_3_RIGHT = 3106
    RIB_4_LEFT = 3107
    RIB_4_RIGHT = 3108
    RIB_5_LEFT = 3109
    RIB_5_RIGHT = 3110
    RIB_6_LEFT = 3111
    RIB_6_RIGHT = 3112
    RIB_7_LEFT = 3113
    RIB_7_RIGHT = 3114
    RIB_8_LEFT = 3115
    RIB_8_RIGHT = 3116
    RIB_9_LEFT = 3117
    RIB_9_RIGHT = 3118
    RIB_10_LEFT = 3119
    RIB_10_RIGHT = 3120
    RIB_11_LEFT = 3121
    RIB_11_RIGHT = 3122
    RIB_12_LEFT = 3123
    RIB_12_RIGHT = 3124

    STERNUM = 3200  # sternum bone

    ## appendicular skeleton
    # upper extremity
    UPPER_EXTREMITY_BOTH = 4000  # whole upper extremity (left+right) without scapula and clavicle
    UPPER_EXTREMITY_LEFT = 4001  # whole upper extremity left without scapula and clavicle
    UPPER_EXTREMITY_RIGHT = 4002  # whole upper extremity right without scapula and clavicle

    # clavicle
    CLAVICLE_BOTH = 4100  # whole clavicle both sides
    CLAVICLE_LEFT = 4101  # whole clavicle left side
    CLAVICLE_RIGHT = 4102  # whole clavicle right side
    CLAVICLE_MEDIAL_LEFT = 4103  # medial part of left clavicle
    CLAVICLE_MEDIAL_RIGHT = 4104  # medial part of right clavicle
    CLAVICLE_LATERAL_LEFT = 4105  # lateral part of left clavicle
    CLAVICLE_LATERAL_RIGHT = 4106  # lateral part of right clavicle

    # scapula
    SCAPULA_BOTH = 4200  # whole scapula both sides
    SCAPULA_LEFT = 4201  # whole scapula left side
    SCAPULA_RIGHT = 4202  # whole scapula right side
    SCAPULA_MEDIAL_LEFT = 4203  # medial part of left scapula
    SCAPULA_MEDIAL_RIGHT = 4204  # medial part of right scapula
    SCAPULA_LATERAL_LEFT = 4205  # lateral part of left scapula
    SCAPULA_LATERAL_RIGHT = 4206  # lateral part of right scapula

    # humerus
    HUMERUS_BOTH = 4300  # whole humerus both sides
    HUMERUS_LEFT = 4301  # whole humerus left side
    HUMERUS_RIGHT = 4302  # whole humerus right side
    HUMERUS_PROXIMAL_LEFT = 4303  # proximal part of left humerus
    HUMERUS_PROXIMAL_RIGHT = 4304  # proximal part of right humerus
    HUMERUS_SHAFT_LEFT = 4305  # shaft part of left humerus
    HUMERUS_SHAFT_RIGHT = 4306  # shaft part of right humerus
    HUMERUS_DISTAL_LEFT = 4307  # distal part of left humerus
    HUMERUS_DISTAL_RIGHT = 4308  # distal part of right humerus

    # ulna
    ULNA_BOTH = 4400  # whole ulna both sides
    ULNA_LEFT = 4401  # whole ulna left side
    ULNA_RIGHT = 4402  # whole ulna right side
    ULNA_PROXIMAL_LEFT = 4403  # proximal part of left ulna
    ULNA_PROXIMAL_RIGHT = 4404  # proximal part of right ulna
    ULNA_SHAFT_LEFT = 4405  # shaft part of left ulna
    ULNA_SHAFT_RIGHT = 4406  # shaft part of right ulna
    ULNA_DISTAL_LEFT = 4407  # distal part of left ulna
    ULNA_DISTAL_RIGHT = 4408  # distal part of right ulna

    # radius
    RADIUS_BOTH = 4500  # whole radius both sides
    RADIUS_LEFT = 4501  # whole radius left side
    RADIUS_RIGHT = 4502  # whole radius right side
    RADIUS_PROXIMAL_LEFT = 4503  # proximal part of left radius
    RADIUS_PROXIMAL_RIGHT = 4504  # proximal part of right radius
    RADIUS_SHAFT_LEFT = 4505  # shaft part of left radius
    RADIUS_SHAFT_RIGHT = 4506  # shaft part of right radius
    RADIUS_DISTAL_LEFT = 4507  # distal part of left radius
    RADIUS_DISTAL_RIGHT = 4508  # distal part of right radius

    # carpals
    CARPALS_BOTH = 4600  # whole carpals both sides
    CARPALS_LEFT = 4601  # whole carpals left side
    CARPALS_RIGHT = 4602  # whole carpals right side

    SCAPHOID_LEFT = 4603  # scaphoid bone of left hand which is part of carpals
    SCAPHOID_RIGHT = 4604  # scaphoid bone of right hand which is part of carpals
    LUNATE_LEFT = 4605  # lunate bone of left hand which is part of carpals
    LUNATE_RIGHT = 4606  # lunate bone of right hand which is part of carpals
    TRIQUETRUM_LEFT = 4607  # triquetrum bone of left hand which is part of carpals
    TRIQUETRUM_RIGHT = 4608  # triquetrum bone of right hand which is part of carpals
    PISIFORM_LEFT = 4609  # pisiform bone of left hand which is part of carpals
    PISIFORM_RIGHT = 4610  # pisiform bone of right hand which is part of carpals
    TRAPEZIUM_LEFT = 4611  # trapezium bone of left hand which is part of carpals
    TRAPEZIUM_RIGHT = 4612  # trapezium bone of right hand which is part of carpals
    TRAPEZOID_LEFT = 4613  # trapezoid bone of left hand which is part of carpals
    TRAPEZOID_RIGHT = 4614  # trapezoid bone of right hand which is part of carpals
    CAPITATE_LEFT = 4615  # capitate bone of left hand which is part of carpals
    CAPITATE_RIGHT = 4616  # capitate bone of right hand which is part of carpals
    HAMATE_LEFT = 4617  # hamate bone of left hand which is part of carpals
    HAMATE_RIGHT = 4618  # hamate bone of right hand which is part of carpals

    # metacarpals
    METACARPALS_BOTH = 4700  # whole metacarpals both sides
    METACARPALS_LEFT = 4701  # whole metacarpals left side
    METACARPALS_RIGHT = 4702  # whole metacarpals right side
    METACARPAL_1_LEFT = 4703  # first metacarpal of left hand which is part of metacarpals
    METACARPAL_1_RIGHT = 4704  # first metacarpal of right hand which is part of metacarpals
    METACARPAL_2_LEFT = 4705  # second metacarpal of left hand which is part of metacarpals
    METACARPAL_2_RIGHT = 4706  # second metacarpal of right hand which is part of metacarpals
    METACARPAL_3_LEFT = 4707  # third metacarpal of left hand which is part of metacarpals
    METACARPAL_3_RIGHT = 4708  # third metacarpal of right hand which is part of metacarpals
    METACARPAL_4_LEFT = 4709  # fourth metacarpal of left hand which is part of metacarpals
    METACARPAL_4_RIGHT = 4710  # fourth metacarpal of right hand which is part of metacarpals
    METACARPAL_5_LEFT = 4711  # fifth metacarpal of left hand which is part of metacarpals
    METACARPAL_5_RIGHT = 4712  # fifth metacarpal of right hand which is part of metacarpals

    # phalanges hand
    PHALANGES_HAND_BOTH = 4800  # whole phalanges of both hands
    PHALANGES_HAND_LEFT = 4801  # whole phalanges of left hand
    PHALANGES_HAND_RIGHT = 4802  # whole phalanges of right hand
    # The first digit after the underscore is the finger number (1-5 from thumb to little finger) and the second digit is the phalanx number (1-2 for thumb, 1-3 for fingers 2-5, from proximal to distal).
    # for example, PHALANGE_HAND_2_3_LEFT corresponds to the distal phalanx of the second finger on the left hand.
    PHALANGE_HAND_1_1_LEFT = 4803  # Thumb (digit 1) has only 2 phalanges: proximal and distal
    PHALANGE_HAND_1_1_RIGHT = 4804
    PHALANGE_HAND_1_2_LEFT = 4805
    PHALANGE_HAND_1_2_RIGHT = 4806
    PHALANGE_HAND_2_1_LEFT = 4807  # Digits 2-5 have 3 phalanges: proximal, middle, and distal
    PHALANGE_HAND_2_1_RIGHT = 4808
    PHALANGE_HAND_2_2_LEFT = 4809
    PHALANGE_HAND_2_2_RIGHT = 4810
    PHALANGE_HAND_2_3_LEFT = 4811
    PHALANGE_HAND_2_3_RIGHT = 4812
    PHALANGE_HAND_3_1_LEFT = 4813
    PHALANGE_HAND_3_1_RIGHT = 4814
    PHALANGE_HAND_3_2_LEFT = 4815
    PHALANGE_HAND_3_2_RIGHT = 4816
    PHALANGE_HAND_3_3_LEFT = 4817
    PHALANGE_HAND_3_3_RIGHT = 4818
    PHALANGE_HAND_4_1_LEFT = 4819
    PHALANGE_HAND_4_1_RIGHT = 4820
    PHALANGE_HAND_4_2_LEFT = 4821
    PHALANGE_HAND_4_2_RIGHT = 4822
    PHALANGE_HAND_4_3_LEFT = 4823
    PHALANGE_HAND_4_3_RIGHT = 4824
    PHALANGE_HAND_5_1_LEFT = 4825
    PHALANGE_HAND_5_1_RIGHT = 4826
    PHALANGE_HAND_5_2_LEFT = 4827
    PHALANGE_HAND_5_2_RIGHT = 4828
    PHALANGE_HAND_5_3_LEFT = 4829
    PHALANGE_HAND_5_3_RIGHT = 4830

    ## lower extremity
    LOWER_EXTREMITY_BOTH = 7000  # whole lower extremity (left+right) excluding sacrum and coccyx
    LOWER_EXTREMITY_LEFT = 7001  # whole lower extremity left
    LOWER_EXTREMITY_RIGHT = 7002  # whole lower extremity right

    # hip
    HIP_BOTH = 7100  # whole hip both sides excluding sacrum and coccyx
    HIP_LEFT = 7101  # hip left side excluding sacrum and coccyx
    HIP_RIGHT = 7102  # hip right side excluding sacrum and coccyx
    ILLIUM_LEFT = 7103  # ilium bone of left hip which is part of hip
    ILLIUM_RIGHT = 7104  # ilium bone of right hip which is part of hip
    ISCHIUM_LEFT = 7105  # ischium bone of left hip which is part of hip
    ISCHIUM_RIGHT = 7106  # ischium bone of right hip which is part of hip
    PUBIS_LEFT = 7107  # pubis bone of left hip which is part of hip
    PUBIS_RIGHT = 7108  # pubis bone of right hip which is part of hip

    # femur
    FEMUR_BOTH = 7200  # whole femur both sides
    FEMUR_LEFT = 7201  # whole femur left side
    FEMUR_RIGHT = 7202  # whole femur right side
    FEMUR_PROXIMAL_LEFT = 7203  # proximal part of left femur
    FEMUR_PROXIMAL_RIGHT = 7204  # proximal part of right femur
    FEMUR_SHAFT_LEFT = 7205  # shaft part of left femur
    FEMUR_SHAFT_RIGHT = 7206  # shaft part of right femur
    FEMUR_DISTAL_LEFT = 7207  # distal part of left femur
    FEMUR_DISTAL_RIGHT = 7208  # distal part of right femur

    # tibia
    TIBIA_BOTH = 7300  # whole tibia both sides
    TIBIA_LEFT = 7301  # whole tibia left side
    TIBIA_RIGHT = 7302  # whole tibia right side
    TIBIA_PROXIMAL_LEFT = 7303  # proximal part of left tibia
    TIBIA_PROXIMAL_RIGHT = 7304  # proximal part of right tibia
    TIBIA_SHAFT_LEFT = 7305  # shaft part of left tibia
    TIBIA_SHAFT_RIGHT = 7306  # shaft part of right tibia
    TIBIA_DISTAL_LEFT = 7307  # distal part of left tibia
    TIBIA_DISTAL_RIGHT = 7308  # distal part of right tibia

    # fibula
    FIBULA_BOTH = 7400  # whole fibula both sides
    FIBULA_LEFT = 7401  # whole fibula left side
    FIBULA_RIGHT = 7402  # whole fibula right side
    FIBULA_PROXIMAL_LEFT = 7403  # proximal part of left fibula
    FIBULA_PROXIMAL_RIGHT = 7404  # proximal part of right fibula
    FIBULA_SHAFT_LEFT = 7405  # shaft part of left fibula
    FIBULA_SHAFT_RIGHT = 7406  # shaft part of right fibula
    FIBULA_DISTAL_LEFT = 7407  # distal part of left fibula
    FIBULA_DISTAL_RIGHT = 7408  # distal part of right fibula

    # patella
    PATELLA_BOTH = 7500  # patella both sides
    PATELLA_LEFT = 7501  # patella left side
    PATELLA_RIGHT = 7502  # patella right side

    # tarsals
    TARSALS_BOTH = 7600  # whole tarsals both sides
    TARSALS_LEFT = 7601  # whole tarsals left side
    TARSALS_RIGHT = 7602  # whole tarsals right side
    TALUS_LEFT = 7603  # talus bone of left foot which is part of tarsals
    TALUS_RIGHT = 7604  # talus bone of right foot which is part of tarsals
    CALCANEUS_LEFT = 7605  # calcaneus bone of left foot which is part of tarsals
    CALCANEUS_RIGHT = 7606  # calcaneus bone of right foot which is part of tarsals
    NAVICULAR_LEFT = 7607  # navicular bone of left foot which is part of tarsals
    NAVICULAR_RIGHT = 7608  # navicular bone of right foot which is part of tarsals
    CUBOID_LEFT = 7609  # cuboid bone of left foot which is part of tarsals
    CUBOID_RIGHT = 7610  # cuboid bone of right foot which is part of tarsals
    LATERAL_CUNEIFORM_LEFT = 7611  # lateral cuneiform bone of left foot which is part of tarsals
    LATERAL_CUNEIFORM_RIGHT = 7612  # lateral cuneiform bone of right foot which is part of tarsals
    INTERMEDIATE_CUNEIFORM_LEFT = 7613  # intermediate cuneiform bone of left foot which is part of tarsals
    INTERMEDIATE_CUNEIFORM_RIGHT = 7614  # intermediate cuneiform bone of right foot which is part of tarsals
    MEDIAL_CUNEIFORM_LEFT = 7615  # medial cuneiform bone of left foot which is part of tarsals
    MEDIAL_CUNEIFORM_RIGHT = 7616  # medial cuneiform bone of right foot which is part of tarsals

    # metatarsals
    METATARSALS_BOTH = 7700  # whole metatarsals both sides
    METATARSALS_LEFT = 7701  # whole metatarsals left side
    METATARSALS_RIGHT = 7702  # whole metatarsals right side
    METATARSAL_1_LEFT = 7703  # first metatarsal of left foot (starting from the big toe) which is part of metatarsals
    METATARSAL_1_RIGHT = 7704  # first metatarsal of right foot (starting from the big toe) which is part of metatarsals
    METATARSAL_2_LEFT = 7705  # second metatarsal of left foot which is part of metatarsals
    METATARSAL_2_RIGHT = 7706  # second metatarsal of right foot which is part of metatarsals
    METATARSAL_3_LEFT = 7707  # third metatarsal of left foot which is part of metatarsals
    METATARSAL_3_RIGHT = 7708  # third metatarsal of right foot which is part of metatarsals
    METATARSAL_4_LEFT = 7709  # fourth metatarsal of left foot which is part of metatarsals
    METATARSAL_4_RIGHT = 7710  # fourth metatarsal of right foot which is part of metatarsals
    METATARSAL_5_LEFT = 7711  # fifth metatarsal of left foot which is part of metatarsals
    METATARSAL_5_RIGHT = 7712  # fifth metatarsal of right foot which is part of metatarsals

    # phalanges foot
    PHALANGES_FOOT_BOTH = 7800
    PHALANGES_FOOT_LEFT = 7801
    PHALANGES_FOOT_RIGHT = 7802
    # The first digit after the underscore is the toe number (1-5 from big toe to little toe) and the second digit is the phalanx number (1-2 for digit 1, 1-3 for digits 2-5, from proximal to distal).
    # for example, PHALANGE_FOOT_2_3_LEFT corresponds to the distal phalanx of the second toe on the left foot.
    PHALANGE_FOOT_1_1_LEFT = 7803  # Big toe (digit 1) has only 2 phalanges: proximal and distal.
    PHALANGE_FOOT_1_1_RIGHT = 7804
    PHALANGE_FOOT_1_2_LEFT = 7805
    PHALANGE_FOOT_1_2_RIGHT = 7806
    PHALANGE_FOOT_2_1_LEFT = 7807  # Toes 2-5 have 3 phalanges: proximal, middle, and distal
    PHALANGE_FOOT_2_1_RIGHT = 7808
    PHALANGE_FOOT_2_2_LEFT = 7809
    PHALANGE_FOOT_2_2_RIGHT = 7810
    PHALANGE_FOOT_2_3_LEFT = 7811
    PHALANGE_FOOT_2_3_RIGHT = 7812
    PHALANGE_FOOT_3_1_LEFT = 7813
    PHALANGE_FOOT_3_1_RIGHT = 7814
    PHALANGE_FOOT_3_2_LEFT = 7815
    PHALANGE_FOOT_3_2_RIGHT = 7816
    PHALANGE_FOOT_3_3_LEFT = 7817
    PHALANGE_FOOT_3_3_RIGHT = 7818
    PHALANGE_FOOT_4_1_LEFT = 7819
    PHALANGE_FOOT_4_1_RIGHT = 7820
    PHALANGE_FOOT_4_2_LEFT = 7821
    PHALANGE_FOOT_4_2_RIGHT = 7822
    PHALANGE_FOOT_4_3_LEFT = 7823
    PHALANGE_FOOT_4_3_RIGHT = 7824
    PHALANGE_FOOT_5_1_LEFT = 7825
    PHALANGE_FOOT_5_1_RIGHT = 7826
    PHALANGE_FOOT_5_2_LEFT = 7827
    PHALANGE_FOOT_5_2_RIGHT = 7828
    PHALANGE_FOOT_5_3_LEFT = 7829
    PHALANGE_FOOT_5_3_RIGHT = 7830

    @classmethod
    def get_names_list(cls):
        return [label.name for label in cls]


bonehub_to_snomed = {  # use https://github.com/ENHANCE-PET/MOOSE/blob/main/moosez/mappings/SNOMED.py
    BoneLabelMap.SKULL.value: {
        "id": 118646007,
        "name": "Entire bone of head",
    },
}
