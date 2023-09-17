package com.code.sample.readmrz


class MRZVerifier {
    // general checking parameters
    private val length36Characters = 36
    private val length44Characters = 44
    val correctStringLength: IntArray = intArrayOf(length36Characters, length44Characters)
    var detectedStringLength: Int = 0
    private val weights = intArrayOf(7, 3, 1)
    private val checkDigitCalculationModulus = 10
    private val issuingStateIndices = intArrayOf(2, 3, 4)
    private val documentNumberIndices = intArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8)
    private val documentNumberCheckDigitIndex = 9
    private val nationalityIndices = intArrayOf(10, 11, 12)
    private val dateOfBirthIndices = intArrayOf(13, 14, 15, 16, 17, 18)
    private val dateOfBirthCheckDigitIndex = 19
    private val sexCharacterIndex = 20
    private val dateOfExpiryIndices = intArrayOf(21, 22, 23, 24, 25, 26)
    private val dateOfExpiryCheckDigitIndex = 27
    private val passportOptionalIndices = intArrayOf(28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41)
    private val passportOptionalCheckDigitIndex = 42
    private val passportCompositeIndices = documentNumberIndices + documentNumberCheckDigitIndex +
            dateOfBirthIndices + dateOfBirthCheckDigitIndex + dateOfExpiryIndices +
            dateOfExpiryCheckDigitIndex + passportOptionalIndices + passportOptionalCheckDigitIndex
    private val passportCompositeCheckDigitIndex = 43
    private val td2OptionalIndices = intArrayOf(28, 29, 30, 31, 32, 33, 34)
    private val td2CompositeIndices = documentNumberIndices + documentNumberCheckDigitIndex +
            dateOfBirthIndices + dateOfBirthCheckDigitIndex + dateOfExpiryIndices +
            dateOfExpiryCheckDigitIndex + td2OptionalIndices
    private val td2CompositeCheckDigitIndex = 35
    private val passportUpperLineFirstCharacter = "P"
    private val visaUpperLineFirstCharacter = "V"
    private val td2UpperLineFirstCharacters = arrayOf("A", "C", "I")
    private val sexCharacters = arrayOf("F", "M", "<")
    private val fillerCharacter = "<"
    private val zeroCharacter = "0"

    // Russia specific checking parameters
    private val rusSpecificPassportNumberIndices = intArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    private val rusSpecificPassportNumberCheckDigitIndex = 12
    private val rusSpecificNationalityIndices = intArrayOf(13, 14, 15)
    private val rusSpecificDateOfBirthIndices = intArrayOf(16, 17, 18, 19, 20, 21)
    private val rusSpecificDateOfBirthCheckDigitIndex = 22
    private val rusSpecificSexCharacterIndex = 23
    private val rusSpecificInvitationNumberIndices = intArrayOf(24, 25, 26, 27, 28, 29, 30, 31, 32)
    private val rusSpecificInvitationNumberCheckDigitIndex = 33
    private val rusSpecificDocumentIDIndices = intArrayOf(34, 35, 36, 37, 38, 39, 40, 41, 42)
    private val rusSpecificDocumentIDCheckDigitIndex = 43
    private val rusSpecificNameIndices = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42)
    private val rusSpecificNameCheckDigitIndex = 43
    private val rusSpecificUpperLineFirstCharacters = arrayOf("A", "V")
    private val rusSpecificSexCharacters = arrayOf("F", "M")

    fun mrzCompositeCheck(textArray: Array<Array<String>>): Boolean {
        var passedCheck = false

        // performing general check
        if ( // checking issuing state
                textArray[0][issuingStateIndices[0]] +
                textArray[0][issuingStateIndices[1]] +
                textArray[0][issuingStateIndices[2]] in countryCodesArray
        ) {
            if ( // checking passport case
                textArray[0][0] == passportUpperLineFirstCharacter &&
                textArray[0].size == length44Characters
            ) {
                // checking optional field
                var optionalFieldPassedCheck = false
                // checking if optional data field is empty
                var optionalFieldIsEmpty = true
                for (i in passportOptionalIndices) {
                    if (textArray[1][i] != fillerCharacter) {
                        optionalFieldIsEmpty = false
                        break
                    }
                }
                // determining if optional field passed the check
                if (
                    optionalFieldIsEmpty &&
                    (textArray[1][passportOptionalCheckDigitIndex] == fillerCharacter ||
                            textArray[1][passportOptionalCheckDigitIndex] == zeroCharacter)
                ) {
                    optionalFieldPassedCheck = true
                } else if (
                    !optionalFieldIsEmpty &&
                    textArray[1][passportOptionalCheckDigitIndex] ==
                    calculateCheckDigit(textArray[1], passportOptionalIndices)
                ) {
                    optionalFieldPassedCheck = true
                }
                // if optional field has passed the check, then checking composite check digit
                if (
                    optionalFieldPassedCheck &&
                    textArray[1][passportCompositeCheckDigitIndex] ==
                    calculateCheckDigit(textArray[1], passportCompositeIndices)
                ) {
                    passedCheck = true
                }
            } else if ( // checking Visa MRV-A and MRV-B case
                textArray[0][0] == visaUpperLineFirstCharacter &&
                (textArray[0].size == length36Characters ||
                        textArray[0].size == length44Characters)
            ) {
                passedCheck = true
            } else if ( // checking TD2 document case
                textArray[0][0] in td2UpperLineFirstCharacters &&
                textArray[0].size == length36Characters &&
                textArray[1][td2CompositeCheckDigitIndex] ==
                calculateCheckDigit(textArray[1], td2CompositeIndices)
            ) {
                passedCheck = true
            }
        }

        // performing Russia specific check
        if (
                !passedCheck &&
                // checking document type
                textArray[0][0] in rusSpecificUpperLineFirstCharacters &&
                // checking length
                textArray[0].size == length44Characters &&
                // checking name
                textArray[0][rusSpecificNameCheckDigitIndex] ==
                calculateCheckDigit(textArray[0], rusSpecificNameIndices)
        ) {
            passedCheck = true
        }

        return passedCheck
    }

    fun mrzString2PreliminaryCheck(textArray: Array<Array<String>>): Boolean {
        var passedCheck = false

        // performing general check
        if (
                // checking document number
                textArray[1][documentNumberCheckDigitIndex] ==
                calculateCheckDigit(textArray[1], documentNumberIndices) &&
                // checking nationality
                textArray[1][nationalityIndices[0]] +
                textArray[1][nationalityIndices[1]] +
                textArray[1][nationalityIndices[2]] in countryCodesArray &&
                // checking date of birth
                textArray[1][dateOfBirthCheckDigitIndex] ==
                calculateCheckDigit(textArray[1], dateOfBirthIndices) &&
                // checking sex
                textArray[1][sexCharacterIndex] in sexCharacters &&
                // checking date of expiry
                textArray[1][dateOfExpiryCheckDigitIndex] ==
                calculateCheckDigit(textArray[1], dateOfExpiryIndices)
        ) {
            passedCheck = true
        }

        // performing Russia specific check
        if (
                !passedCheck &&
                // checking length
                textArray[1].size == length44Characters &&
                // checking the number of passport, for which this document is issued
                textArray[1][rusSpecificPassportNumberCheckDigitIndex] ==
                calculateCheckDigit(textArray[1], rusSpecificPassportNumberIndices) &&
                // checking nationality
                textArray[1][rusSpecificNationalityIndices[0]] +
                textArray[1][rusSpecificNationalityIndices[1]] +
                textArray[1][rusSpecificNationalityIndices[2]] in countryCodesArray &&
                // checking date of birth
                textArray[1][rusSpecificDateOfBirthCheckDigitIndex] ==
                calculateCheckDigit(textArray[1], rusSpecificDateOfBirthIndices) &&
                // checking sex
                textArray[1][rusSpecificSexCharacterIndex] in rusSpecificSexCharacters &&
                // checking invitation number
                textArray[1][rusSpecificInvitationNumberCheckDigitIndex] ==
                calculateCheckDigit(textArray[1], rusSpecificInvitationNumberIndices) &&
                // checking this document ID
                textArray[1][rusSpecificDocumentIDCheckDigitIndex] ==
                calculateCheckDigit(textArray[1], rusSpecificDocumentIDIndices)
        ) {
            passedCheck = true
        }

        return passedCheck
    }

    private fun calculateCheckDigit(lineArray: Array<String>, characterIndices: IntArray): String {
        var sum = 0
        for (i in characterIndices.indices) {
            sum += characterToValueMap[lineArray[characterIndices[i]]]!! * weights[i % weights.size]
        }
        return (sum % checkDigitCalculationModulus).toString()
    }

    private val characterToValueMap = mapOf<String, Int>(
        "0" to 0,
        "1" to 1,
        "2" to 2,
        "3" to 3,
        "4" to 4,
        "5" to 5,
        "6" to 6,
        "7" to 7,
        "8" to 8,
        "9" to 9,

        "<" to 0,

        "A" to 10,
        "B" to 11,
        "C" to 12,
        "D" to 13,
        "E" to 14,
        "F" to 15,
        "G" to 16,
        "H" to 17,
        "I" to 18,
        "J" to 19,
        "K" to 20,
        "L" to 21,
        "M" to 22,
        "N" to 23,
        "O" to 24,
        "P" to 25,
        "Q" to 26,
        "R" to 27,
        "S" to 28,
        "T" to 29,
        "U" to 30,
        "V" to 31,
        "W" to 32,
        "X" to 33,
        "Y" to 34,
        "Z" to 35
    )

    private val countryCodesArray = arrayOf<String>(
        // general country codes
        "AFG", "ALB", "DZA", "ASM", "AND", "AGO", "AIA", "ATA", "ATG", "ARG", "ARM", "ABW",
        "AUS", "AUT", "AZE", "BHS", "BHR", "BGD", "BRB", "BLR", "BEL", "BLZ", "BEN", "BMU",
        "ALA", "BTN", "BOL", "BES", "BIH", "BWA", "BVT", "BRA", "IOT", "BRN", "BGR", "BFA",
        "BDI", "CPV", "KHM", "CMR", "CAN", "CYM", "CAF", "TCD", "CHL", "CHN", "CXR", "CCK",
        "COL", "COM", "COD", "COG", "COK", "CRI", "HRV", "CUB", "CUW", "CYP", "CZE", "CIV",
        "DNK", "DJI", "DMA", "DOM", "ECU", "EGY", "SLV", "GNQ", "ERI", "EST", "SWZ", "ETH",
        "FLK", "FRO", "FJI", "FIN", "FRA", "GUF", "PYF", "ATF", "GAB", "GMB", "GEO", "DEU",
        "GHA", "GIB", "GRC", "GRL", "GRD", "GLP", "GUM", "GTM", "GGY", "GIN", "GNB", "GUY",
        "HTI", "HMD", "VAT", "HND", "HKG", "HUN", "ISL", "IND", "IDN", "IRN", "IRQ", "IRL",
        "IMN", "ISR", "ITA", "JAM", "JPN", "JEY", "JOR", "KAZ", "KEN", "KIR", "PRK", "KOR",
        "KWT", "KGZ", "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU", "LUX", "MAC",
        "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MHL", "MTQ", "MRT", "MUS", "MYT", "MEX",
        "FSM", "MDA", "MCO", "MNG", "MNE", "MSR", "MAR", "MOZ", "MMR", "NAM", "NRU", "NPL",
        "NLD", "NCL", "NZL", "NIC", "NER", "NGA", "NIU", "NFK", "MKD", "MNP", "NOR", "OMN",
        "PAK", "PLW", "PSE", "PAN", "PNG", "PRY", "PER", "PHL", "PCN", "POL", "PRT", "PRI",
        "QAT", "ROU", "RUS", "RWA", "REU", "BLM", "SHN", "KNA", "LCA", "MAF", "SPM", "VCT",
        "WSM", "SMR", "STP", "SAU", "SEN", "SRB", "SYC", "SLE", "SGP", "SXM", "SVK", "SVN",
        "SLB", "SOM", "ZAF", "SGS", "SSD", "ESP", "LKA", "SDN", "SUR", "SJM", "SWE", "CHE",
        "SYR", "TWN", "TJK", "TZA", "THA", "TLS", "TGO", "TKL", "TON", "TTO", "TUN", "TKM",
        "TCA", "TUV", "TUR", "UGA", "UKR", "ARE", "GBR", "UMI", "USA", "URY", "UZB", "VUT",
        "VEN", "VNM", "VGB", "VIR", "WLF", "ESH", "YEM", "ZMB", "ZWE",

        // special codes
        "GBD", "GBN", "GBO", "GBS", "GBP", "RKS", "EUE", "UNO", "UNA", "UNK", "XBA", "XIM",
        "XCC", "XCE", "XCO", "XEC", "XPO", "XES", "XOM", "XDC", "XXA", "XXB", "XXC", "XXX",
        "ANT", "NTZ", "UTO",

        // additional codes
        "YUG",
        "SP<", "TM<", "D<<"
    )
}


