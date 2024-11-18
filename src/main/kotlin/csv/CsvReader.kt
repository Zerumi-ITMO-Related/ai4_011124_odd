package io.github.zerumi.csv

import io.github.zerumi.model.Diabetes
import java.io.InputStream

data class Dataset(
    val labels: List<String>,
    val values: List<Diabetes>,
)

fun readCsv(inputStream: InputStream): Dataset {
    val reader = inputStream.bufferedReader()
    val header = reader.readLine().split(',')

    return Dataset(header, reader.lineSequence().filter { it.isNotBlank() }.map {
        val (pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, pedigree, age, outcome) = it.split(
            ','
        )
        Diabetes(
            pregnancies.toFloat(),
            glucose.toFloat(),
            bloodPressure.toFloat(),
            skinThickness.toFloat(),
            insulin.toFloat(),
            bmi.toFloat(),
            pedigree.toFloat(),
            age.toFloat(),
            outcome.toFloat(),
        )
    }.toList()
    )
}

private operator fun <E> List<E>.component6(): E = this[5]

private operator fun <E> List<E>.component7(): E = this[6]

private operator fun <E> List<E>.component8(): E = this[7]

private operator fun <E> List<E>.component9(): E = this[8]
