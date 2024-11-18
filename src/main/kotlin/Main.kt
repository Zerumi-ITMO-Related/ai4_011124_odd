package io.github.zerumi

import io.github.zerumi.csv.readCsv
import io.github.zerumi.model.*
import io.github.zerumi.stat.ListStat
import io.github.zerumi.stat.knn
import io.github.zerumi.stat.normalized
import io.github.zerumi.util.head
import kotlin.math.roundToInt
import kotlin.system.exitProcess

fun main() {
    val dataset = readCsv(
        {}::class.java.classLoader.getResourceAsStream("diabetes.csv")
            ?: throw IllegalArgumentException("Resource not found")
    )

    val datasetSize = dataset.values.size

    val pregnanciesStat = ListStat.fromList(dataset.labels[0], dataset.values.extractPregnancies())
    val glucoseStat = ListStat.fromList(dataset.labels[1], dataset.values.extractGlucose())
    val bloodPressureStat = ListStat.fromList(dataset.labels[2], dataset.values.extractBloodPressure())
    val skinThicknessStat = ListStat.fromList(dataset.labels[3], dataset.values.extractSkinThickness())
    val insulinStat = ListStat.fromList(dataset.labels[4], dataset.values.extractInsulin())
    val bmiStat = ListStat.fromList(dataset.labels[5], dataset.values.extractBMI())
    val pedigreeStat = ListStat.fromList(dataset.labels[6], dataset.values.extractPedigree())
    val ageStat = ListStat.fromList(dataset.labels[7], dataset.values.extractAge())
    val outcomeStat = ListStat.fromList(dataset.labels[8], dataset.values.extractOutcome())

    println(pregnanciesStat)
    println(glucoseStat)
    println(bloodPressureStat)
    println(skinThicknessStat)
    println(insulinStat)
    println(bmiStat)
    println(pedigreeStat)
    println(ageStat)
    println(outcomeStat)

    println("----------------------------")

    val pregnanciesNormalized = dataset.values.extractPregnancies().normalized()
    val glucoseNormalized = dataset.values.extractGlucose().normalized()
    val bloodPressureNormalized = dataset.values.extractBloodPressure().normalized()
    val skinThicknessNormalized = dataset.values.extractSkinThickness().normalized()
    val insulinNormalized = dataset.values.extractInsulin().normalized()
    val bmiNormalized = dataset.values.extractBMI().normalized()
    val pedigreeNormalized = dataset.values.extractPedigree().normalized()
    val ageNormalized = dataset.values.extractAge().normalized()
    val outcomeNormalized = dataset.values.extractOutcome().normalized()

    println("${dataset.labels[0]} normalized: ${pregnanciesNormalized.head()}")
    println("${dataset.labels[1]} normalized: ${glucoseNormalized.head()}")
    println("${dataset.labels[2]} normalized: ${bloodPressureNormalized.head()}")
    println("${dataset.labels[3]} normalized: ${skinThicknessNormalized.head()}")
    println("${dataset.labels[4]} normalized: ${bmiNormalized.head()}")
    println("${dataset.labels[5]} normalized: ${pedigreeNormalized.head()}")
    println("${dataset.labels[6]} normalized: ${insulinNormalized.head()}")
    println("${dataset.labels[7]} normalized: ${ageNormalized.head()}")
    println("${dataset.labels[8]} normalized: ${outcomeNormalized.head()}")

    println("----------------------------")

    val normalizedDataset = List(datasetSize) {
        Diabetes(
            pregnancies = pregnanciesNormalized[it],
            glucose = glucoseNormalized[it],
            bloodPressure = bloodPressureNormalized[it],
            skinThickness = skinThicknessNormalized[it],
            insulin = insulinNormalized[it],
            bmi = bmiNormalized[it],
            pedigree = pedigreeNormalized[it],
            age = ageNormalized[it],
            outcome = outcomeNormalized[it],
        )
    }

    val k = 5

    val testDataSize = (datasetSize * 0.6).roundToInt()
    val shuffledNormalizedDataset = normalizedDataset.shuffled()

    val testData = shuffledNormalizedDataset.take(testDataSize)
    val targetData = shuffledNormalizedDataset.takeLast(datasetSize - testDataSize)

    println("Applying KNN (k = $k)...")

    val knnResults = knn(testData, targetData, k)

    println("Compare head outcomes: ${knnResults.head().map { it.outcomeResult }} vs real ${
        targetData.head().map { it.outcome }
    }")

    while (true) {
        println("Available ${targetData.size} results from 0 to ${targetData.size - 1}")
        print("Enter index to see the result (or -1 to exit): ")

        val index = readln().toIntOrNull() ?: exitProcess(0)

        if (index !in 0..targetData.lastIndex) exitProcess(0)

        println("Object: ${targetData[index]}")
        println("Object outcome: ${targetData[index].outcome}")
        println("KNN result for this object: ${knnResults[index].outcomeResult}")
        println("Neighbours: ${knnResults[index].neighbours}")

        println("----------------------------")
    }
}
