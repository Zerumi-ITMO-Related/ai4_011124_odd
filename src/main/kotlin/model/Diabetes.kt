package io.github.zerumi.model

// Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,Pedigree,Age,Outcome
data class Diabetes(
    val pregnancies: Float,
    val glucose: Float,
    val bloodPressure: Float,
    val skinThickness: Float,
    val insulin: Float,
    val bmi: Float,
    val pedigree: Float,
    val age: Float,
    val outcome: Float
) {
    operator fun minus(target: Diabetes) = Diabetes(
        pregnancies = this.pregnancies - target.pregnancies,
        glucose = this.glucose - target.glucose,
        bloodPressure = this.bloodPressure - target.bloodPressure,
        skinThickness = this.skinThickness - target.skinThickness,
        insulin = this.insulin - target.insulin,
        bmi = this.bmi - target.bmi,
        pedigree = this.pedigree - target.pedigree,
        age = this.age - target.age,
        outcome = this.outcome - target.outcome,
    )

    operator fun times(target: Diabetes) = Diabetes(
        pregnancies = this.pregnancies * target.pregnancies,
        glucose = this.glucose * target.glucose,
        bloodPressure = this.bloodPressure * target.bloodPressure,
        skinThickness = this.skinThickness * target.skinThickness,
        insulin = this.insulin * target.insulin,
        bmi = this.bmi * target.bmi,
        pedigree = this.pedigree * target.pedigree,
        age = this.age * target.age,
        outcome = this.outcome * target.outcome,
    )

    fun sum() =
        this.pregnancies + this.glucose + this.bloodPressure + this.skinThickness + this.insulin + this.bmi + this.pedigree + this.age
}

fun List<Diabetes>.extractPregnancies() = this.map { it.pregnancies }

fun List<Diabetes>.extractGlucose() = this.map { it.glucose }

fun List<Diabetes>.extractBloodPressure() = this.map { it.bloodPressure }

fun List<Diabetes>.extractSkinThickness() = this.map { it.skinThickness }

fun List<Diabetes>.extractInsulin() = this.map { it.insulin }

fun List<Diabetes>.extractBMI() = this.map { it.bmi }

fun List<Diabetes>.extractPedigree() = this.map { it.pedigree }

fun List<Diabetes>.extractAge() = this.map { it.age }

fun List<Diabetes>.extractOutcome() = this.map { it.outcome }
