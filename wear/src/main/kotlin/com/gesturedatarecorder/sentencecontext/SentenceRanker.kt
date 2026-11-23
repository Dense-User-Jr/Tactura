package com.gesturedatarecorder.sentencecontext

import android.content.Context
import android.util.Log

/** Ranks lexicon matches using bundled corpus counts and sentences finished in this session. */
internal class SentenceRanker private constructor(private val counts: Map<String, Map<String, Int>>) {
    private val confirmed = mutableMapOf<String, MutableMap<String, Int>>()

    @Synchronized
    fun learnFinishedSentence(words: List<String>) {
        for (index in 1 until words.size) {
            val word = words[index]
            val contexts = listOf(words[index - 1], words.subList(maxOf(0, index - 2), index).joinToString(" "))
            for (context in contexts.distinct()) {
                val continuations = confirmed.getOrPut(context) { mutableMapOf() }
                continuations[word] = continuations.getOrElse(word) { 0 } + 1
            }
        }
    }

    @Synchronized
    fun rank(prefix: List<String>, candidates: List<String>, useCorpus: Boolean = true): List<String> {
        if (prefix.isEmpty() || candidates.size < 2) return candidates
        val last = prefix.last()
        val previousPair = prefix.takeLast(2).takeIf { it.size == 2 }?.joinToString(" ")
        val bigram = if (useCorpus) counts[last].orEmpty() else emptyMap()
        val trigram = if (useCorpus) counts[previousPair].orEmpty() else emptyMap()
        val ranked = candidates.sortedByDescending { word ->
            bigram.getOrElse(word) { 0 } + TRIGRAM_WEIGHT * trigram.getOrElse(word) { 0 }
        }
        val contexts = listOfNotNull(previousPair, last)
        val preferred = contexts
            .map { context -> preferences[context].orEmpty().filter { it in candidates } }
            .firstOrNull { it.isNotEmpty() }.orEmpty()
        val baseline = preferred + ranked.filterNot { it in preferred }
        if (!useCorpus) return baseline
        val learned = contexts.asSequence()
            .mapNotNull { confirmed[it] }
            .firstOrNull { options -> candidates.any { it in options } }
            ?: return baseline
        return baseline.sortedByDescending { learned[it] ?: 0 }
    }

    companion object {
        private const val TAG = "SentenceRanker"
        private const val ASSET = "sentence_context.tsv"
        private const val TRIGRAM_WEIGHT = 4

        // Keep established demo phrases available even when sparsely represented in the corpus.
        private val preferences = mapOf(
            "rent is" to listOf("due"),
            "this is" to listOf("too"),
            "it is" to listOf("too"),
            "need" to listOf("to"),
            "in" to listOf("two", "the")
        )

        val fallback get() = SentenceRanker(emptyMap())

        fun load(context: Context): SentenceRanker = try {
            context.assets.open(ASSET).bufferedReader().useLines { parse(it) }
        } catch (error: java.io.IOException) {
            Log.e(TAG, "Could not load sentence context", error)
            fallback
        }

        internal fun parse(lines: Sequence<String>): SentenceRanker {
            val counts = mutableMapOf<String, MutableMap<String, Int>>()
            for (line in lines) {
                val cells = line.split('\t')
                if (cells.size != 3) continue
                val count = cells[2].toIntOrNull()?.takeIf { it > 0 } ?: continue
                counts.getOrPut(cells[0]) { mutableMapOf() }[cells[1]] = count
            }
            return SentenceRanker(counts)
        }
    }
}
