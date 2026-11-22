package com.gesturedatarecorder.sentencecontext

import com.gesturedatarecorder.recording.Gesture
import com.gesturedatarecorder.recording.Lexicon
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import java.io.File

class SentenceRankerTest {
    @Test
    fun contextPromotesHiddenMatchesFromBundledLexicon() {
        val asset = listOf("src/main/assets/lexicon.csv", "wear/src/main/assets/lexicon.csv")
            .map(::File)
            .first { it.isFile }
        val entries = asset.useLines { Lexicon.parse(it, Gesture.entries) }
        val matches = Lexicon.exact(entries, "94")

        assertEquals(listOf("to", "do", "two"), matches.take(3))
        assertEquals("due", SentenceRanker.fallback.rank(listOf("the", "rent", "is"), matches).first())
        assertEquals("too", SentenceRanker.fallback.rank(listOf("this", "is"), matches).first())
        assertEquals("two", SentenceRanker.fallback.rank(listOf("in"), matches).first())
    }

    @Test
    fun bundledCountsRankAnUnscriptedContext() {
        val asset = listOf("src/main/assets/sentence_context.tsv", "wear/src/main/assets/sentence_context.tsv")
            .map(::File)
            .first { it.isFile }
        val ranker = asset.useLines { SentenceRanker.parse(it) }
        val choices = listOf("to", "do", "two", "too", "due")
        assertEquals("do", ranker.rank(listOf("didn't"), choices).first())
        assertEquals(choices, ranker.rank(listOf("didn't"), choices, useCorpus = false))
        assertEquals(choices, ranker.rank(listOf("unseen"), choices))
    }

    @Test
    fun absentContextOrUnavailablePreferencePreservesLexiconOrder() {
        val matches = listOf("to", "do", "two")

        assertEquals(matches, SentenceRanker.fallback.rank(emptyList(), matches))
        assertEquals(matches, SentenceRanker.fallback.rank(listOf("is"), matches))
        assertEquals(matches, SentenceRanker.fallback.rank(listOf("rent", "is"), matches))
        assertFalse("due" in SentenceRanker.fallback.rank(listOf("rent", "is"), matches))
    }
}
