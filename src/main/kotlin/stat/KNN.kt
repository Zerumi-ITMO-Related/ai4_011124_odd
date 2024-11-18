package io.github.zerumi.stat

import io.github.zerumi.model.Diabetes
import kotlin.math.sqrt

data class KNNResult(
    val diabetes: Diabetes,
    val neighbours: List<Pair<Float, Float>>,
    val outcomeResult: Float,
)

// all data should be normalized
fun euclideanDistance(source: Diabetes, target: Diabetes) = sqrt(((source - target) * (source - target)).sum())

fun knn(testData: List<Diabetes>, targetData: List<Diabetes>, k: Int): List<KNNResult> = targetData.map { target ->
    KNNResult(
        diabetes = target,
        neighbours = testData.asSequence().map { Pair(it.outcome, euclideanDistance(it, target)) }
            .sortedBy { it.second }.take(k).toList(),

        outcomeResult = testData.asSequence().map { Pair(it.outcome, euclideanDistance(it, target)) }
            .sortedBy { it.second }.take(k).map { it.first }.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key
            ?: throw RuntimeException()
    )
}
