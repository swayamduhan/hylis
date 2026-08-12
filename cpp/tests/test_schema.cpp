// Tests for the typed column layer: logical types, values, and the schema.
//
// This is the layer that decides *which structures are candidates at all*, so
// most of what follows is that filter stated in various forms — a string
// column must never be offered a learned index, a duplicated column must never
// be offered a structure that maps one key to one row, and a value that does
// not parse must be refused at the boundary rather than skipped somewhere the
// caller cannot see.
//
// The round-trip tests matter more than they look. A schema that cannot render
// what it parsed cannot write a checkpoint, and a timestamp that survives a
// save/load only approximately is a data-loss bug that hides for months.

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

#include "index/logical_type.hpp"
#include "query/schema.hpp"
#include "storage/record.hpp"

using hylis::index::Datum;
using hylis::index::KeyEncoding;
using hylis::index::LogicalType;
using hylis::index::datum_equal;
using hylis::index::datum_less;
using hylis::index::format_datum;
using hylis::index::key_encoding_from_string;
using hylis::index::logical_type_from_string;
using hylis::index::parse_datum;
using hylis::index::prefix_upper_bound;
using hylis::index::to_string;
using hylis::index::try_parse_datum;
using hylis::index::type_supports_rmi;
using hylis::query::ColumnDef;
using hylis::query::Schema;
using hylis::storage::Record;

// ---------------------------------------------------------------------------
// Logical types
// ---------------------------------------------------------------------------

TEST(LogicalTypes, EveryNameRoundTrips) {
    for (LogicalType t : {LogicalType::Int64, LogicalType::Double,
                          LogicalType::String, LogicalType::Bool,
                          LogicalType::Timestamp, LogicalType::Vector}) {
        EXPECT_EQ(logical_type_from_string(to_string(t)), t);
    }
    EXPECT_THROW(logical_type_from_string("decimal"), std::invalid_argument);
}

TEST(LogicalTypes, EveryEncodingNameRoundTrips) {
    for (KeyEncoding e : {KeyEncoding::Native, KeyEncoding::Composite,
                          KeyEncoding::Position, KeyEncoding::Dictionary}) {
        EXPECT_EQ(key_encoding_from_string(to_string(e)), e);
    }
}

// The single most load-bearing fact in this whole layer. RMIndex fits models
// to static_cast<double>(key); a string has no such cast that preserves an
// ordering the model did not itself impose.
TEST(LogicalTypes, OnlyNumericTypesCanCarryALearnedIndex) {
    EXPECT_TRUE(type_supports_rmi(LogicalType::Int64));
    EXPECT_TRUE(type_supports_rmi(LogicalType::Double));
    EXPECT_TRUE(type_supports_rmi(LogicalType::Timestamp));
    EXPECT_FALSE(type_supports_rmi(LogicalType::String));
    EXPECT_FALSE(type_supports_rmi(LogicalType::Bool));
    EXPECT_FALSE(type_supports_rmi(LogicalType::Vector));
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

TEST(Parsing, TrailingJunkIsAFailureNotAValue) {
    // "12abc" is a data-entry mistake. Accepting it as 12 would turn a typo
    // into a silently wrong index, which is the failure mode with no symptom.
    Datum out;
    EXPECT_FALSE(try_parse_datum(LogicalType::Int64, "12abc", &out));
    EXPECT_FALSE(try_parse_datum(LogicalType::Int64, "", &out));
    EXPECT_FALSE(try_parse_datum(LogicalType::Int64, "1.5", &out));
    EXPECT_FALSE(try_parse_datum(LogicalType::Double, "4.2kg", &out));
    EXPECT_TRUE(try_parse_datum(LogicalType::Int64, "-42", &out));
    EXPECT_EQ(std::get<std::int64_t>(out), -42);
}

TEST(Parsing, NaNAndInfinityAreRefusedAtTheBoundary) {
    // NaN has no ordering, so a single one would corrupt every structure here
    // — a sorted array containing it is not sorted under any comparison.
    // Infinities defeat the mean-centred fit the linear models depend on.
    Datum out;
    EXPECT_FALSE(try_parse_datum(LogicalType::Double, "nan", &out));
    EXPECT_FALSE(try_parse_datum(LogicalType::Double, "NaN", &out));
    EXPECT_FALSE(try_parse_datum(LogicalType::Double, "inf", &out));
    EXPECT_FALSE(try_parse_datum(LogicalType::Double, "-inf", &out));
    EXPECT_TRUE(try_parse_datum(LogicalType::Double, "1e30", &out));
}

TEST(Parsing, BoolAcceptsTheFourSpellingsAndNothingElse) {
    Datum out;
    for (const char* yes : {"true", "TRUE", "True", "1"}) {
        ASSERT_TRUE(try_parse_datum(LogicalType::Bool, yes, &out)) << yes;
        EXPECT_TRUE(std::get<bool>(out));
    }
    for (const char* no : {"false", "FALSE", "0"}) {
        ASSERT_TRUE(try_parse_datum(LogicalType::Bool, no, &out)) << no;
        EXPECT_FALSE(std::get<bool>(out));
    }
    EXPECT_FALSE(try_parse_datum(LogicalType::Bool, "yes", &out));
    EXPECT_FALSE(try_parse_datum(LogicalType::Bool, "", &out));
}

TEST(Parsing, StringsAreTakenExactly) {
    // No trimming, no case folding, no interpretation. The B+ tree orders by
    // bytes, so anything done here would have to be undone to explain a query.
    EXPECT_EQ(std::get<std::string>(parse_datum(LogicalType::String, "  Nike ")),
              "  Nike ");
    EXPECT_EQ(std::get<std::string>(parse_datum(LogicalType::String, "")), "");
}

TEST(Timestamps, IsoAndEpochBothParseAndNormaliseToIso) {
    const auto round = [](const std::string& in) {
        return format_datum(LogicalType::Timestamp,
                            parse_datum(LogicalType::Timestamp, in));
    };
    EXPECT_EQ(round("2026-08-13T14:30:00Z"), "2026-08-13T14:30:00Z");
    EXPECT_EQ(round("2026-08-13 14:30:00"), "2026-08-13T14:30:00Z");
    EXPECT_EQ(round("2026-08-13"), "2026-08-13T00:00:00Z");
    EXPECT_EQ(round("2026-08-13T14:30:00.250Z"), "2026-08-13T14:30:00.250Z");
    // Epoch millis are accepted and normalise to the canonical text form, so
    // a checkpoint is diffable whichever way the value arrived.
    EXPECT_EQ(round("0"), "1970-01-01T00:00:00Z");
}

TEST(Timestamps, DatesBeforeTheEpochDoNotRoundTheWrongWay) {
    // Negative epoch millis with a non-zero time of day are where integer
    // division toward zero silently produces the previous day.
    const auto round = [](const std::string& in) {
        return format_datum(LogicalType::Timestamp,
                            parse_datum(LogicalType::Timestamp, in));
    };
    EXPECT_EQ(round("1900-01-01T00:00:00Z"), "1900-01-01T00:00:00Z");
    EXPECT_EQ(round("1969-12-31T23:59:59Z"), "1969-12-31T23:59:59Z");
    EXPECT_EQ(round("1600-02-29T12:00:00Z"), "1600-02-29T12:00:00Z");
}

TEST(Timestamps, MalformedInputIsRefused) {
    Datum out;
    EXPECT_FALSE(try_parse_datum(LogicalType::Timestamp, "2026-13-01", &out));
    EXPECT_FALSE(try_parse_datum(LogicalType::Timestamp, "2026-08-32", &out));
    EXPECT_FALSE(try_parse_datum(LogicalType::Timestamp, "not-a-date", &out));
    // An offset other than Z is refused rather than quietly treated as UTC: a
    // timestamp wrong by hours is worse than one that failed to load.
    EXPECT_FALSE(try_parse_datum(LogicalType::Timestamp,
                                 "2026-08-13T14:30:00+05:30", &out));
}

TEST(Formatting, DoublesRoundTripThroughTheirTextForm) {
    // Six significant digits, the default, would lose the low bits of a
    // coordinate or a price. Seventeen is what round-trips every double.
    for (double v : {0.1, 1.0 / 3.0, 1e-300, 1.7976931348623157e308,
                     -2.5e-17, 123456789.123456789}) {
        const std::string text = format_datum(LogicalType::Double, Datum{v});
        EXPECT_EQ(std::get<double>(parse_datum(LogicalType::Double, text)), v)
            << text;
    }
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

TEST(Comparison, MixedTypesThrowRatherThanRankByAlternative) {
    // std::variant's own operator< compares the alternative index first, which
    // would rank every integer below every string and look entirely plausible
    // in a result set. A type mismatch is a bug at the call site and has to
    // say so.
    EXPECT_THROW(datum_less(Datum{std::int64_t{1}}, Datum{std::string("a")}),
                 std::invalid_argument);
    EXPECT_THROW(datum_equal(Datum{1.5}, Datum{std::int64_t{1}}),
                 std::invalid_argument);
}

TEST(Comparison, StringsOrderByBytesNotByNumericValue) {
    // The consequence a user sees: "10" sorts before "9" in a string column
    // and after it in an integer one. Which is why ColumnInfo reports the type
    // rather than leaving it to be inferred.
    EXPECT_TRUE(datum_less(Datum{std::string("10")}, Datum{std::string("9")}));
    EXPECT_TRUE(datum_less(Datum{std::int64_t{9}}, Datum{std::int64_t{10}}));
}

// ---------------------------------------------------------------------------
// Prefixes
// ---------------------------------------------------------------------------

TEST(Prefixes, TheUpperBoundIsTheSuccessorOfTheLastByte) {
    std::string upper;
    ASSERT_TRUE(prefix_upper_bound("nike", &upper));
    EXPECT_EQ(upper, "nikf");
}

TEST(Prefixes, CarryPropagatesThroughTrailingMaxBytes) {
    std::string p = "a";
    p += static_cast<char>(0xFF);
    p += static_cast<char>(0xFF);
    std::string upper;
    ASSERT_TRUE(prefix_upper_bound(p, &upper));
    EXPECT_EQ(upper, "b");
}

TEST(Prefixes, AnAllMaxPrefixHasNoUpperBound) {
    // Every string beginning with 0xFF 0xFF sorts above it and there is no
    // successor to stop at, so the caller has to fall back to a >= scan. The
    // function says so rather than returning a wrong bound.
    std::string p;
    p += static_cast<char>(0xFF);
    p += static_cast<char>(0xFF);
    std::string upper;
    EXPECT_FALSE(prefix_upper_bound(p, &upper));
    EXPECT_FALSE(prefix_upper_bound("", &upper));
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

Schema shop_schema() {
    Schema s;
    s.add(ColumnDef("price", LogicalType::Int64));
    s.add(ColumnDef("weight", LogicalType::Double));
    s.add(ColumnDef("category", LogicalType::String));
    s.add(ColumnDef("in_stock", LogicalType::Bool));
    s.add(ColumnDef("created_at", LogicalType::Timestamp));
    s.add(ColumnDef("embedding", LogicalType::Vector, 128));
    return s;
}

TEST(SchemaBasics, ColumnsAreLookedUpByNameAndTypeIsReported) {
    const Schema s = shop_schema();
    EXPECT_EQ(s.size(), 6u);
    EXPECT_TRUE(s.has("price"));
    EXPECT_FALSE(s.has("pirce"));
    EXPECT_EQ(s.type_of("category"), LogicalType::String);
    EXPECT_EQ(s.column("embedding").dim, 128u);
    EXPECT_THROW(s.column("pirce"), std::invalid_argument);
}

TEST(SchemaBasics, ScalarAndVectorColumnsAreSeparable) {
    const Schema s = shop_schema();
    EXPECT_EQ(s.scalar_columns().size(), 5u);
    ASSERT_EQ(s.vector_columns().size(), 1u);
    EXPECT_EQ(s.vector_columns().front(), "embedding");
}

TEST(SchemaBasics, ADuplicateColumnIsRefused) {
    Schema s;
    s.add(ColumnDef("price", LogicalType::Int64));
    EXPECT_THROW(s.add(ColumnDef("price", LogicalType::Double)),
                 std::invalid_argument);
}

TEST(SchemaBasics, AVectorColumnNeedsADimensionAndNothingElseMayHaveOne) {
    Schema s;
    EXPECT_THROW(s.add(ColumnDef("e", LogicalType::Vector, 0)),
                 std::invalid_argument);
    EXPECT_THROW(s.add(ColumnDef("price", LogicalType::Int64, 8)),
                 std::invalid_argument);
    EXPECT_NO_THROW(s.add(ColumnDef("e", LogicalType::Vector, 64)));
}

// The whole point of having a schema rather than a convention.
TEST(SchemaEnforcement, AnUnknownColumnIsAnError) {
    const Schema s = shop_schema();
    Record typo(1, {{"pirce", "40"}});
    EXPECT_FALSE(s.accepts(typo));
    EXPECT_THROW(s.validate(typo), std::invalid_argument);
}

TEST(SchemaEnforcement, AValueOfTheWrongTypeIsAnError) {
    const Schema s = shop_schema();
    EXPECT_FALSE(s.accepts(Record(1, {{"price", "abc"}})));
    EXPECT_FALSE(s.accepts(Record(1, {{"in_stock", "maybe"}})));
    EXPECT_FALSE(s.accepts(Record(1, {{"created_at", "yesterday"}})));
    EXPECT_TRUE(s.accepts(Record(1, {{"price", "4000"},
                                     {"category", "shoes"},
                                     {"in_stock", "true"},
                                     {"created_at", "2026-08-13"}})));
}

TEST(SchemaEnforcement, AMissingColumnIsNotAnError) {
    // Asymmetric on purpose: an absent value is a row that matches no
    // predicate on that column, which is close enough to SQL's three-valued
    // logic and much simpler than a third truth value.
    const Schema s = shop_schema();
    EXPECT_TRUE(s.accepts(Record(1, {{"price", "4000"}})));
    EXPECT_TRUE(s.accepts(Record(2, {})));
}

TEST(SchemaEnforcement, AnEmbeddingInTheRecordPayloadIsRefused) {
    // A 128-float vector is ~700 bytes of base64 per row. Allowing it would
    // make the write-ahead log the dominant cost of the whole system, so the
    // refusal names the alternative rather than just saying no.
    const Schema s = shop_schema();
    Record r(1, {{"embedding", "0.1,0.2,0.3"}});
    EXPECT_THROW(s.validate(r), std::invalid_argument);
}

TEST(SchemaEnforcement, TheErrorNamesTheColumnAndTheDeclaredColumns) {
    const Schema s = shop_schema();
    try {
        s.validate(Record(7, {{"pirce", "40"}}));
        FAIL() << "expected a throw";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("pirce"), std::string::npos);
        EXPECT_NE(what.find("price"), std::string::npos) << what;
        EXPECT_NE(what.find('7'), std::string::npos) << what;
    }
}

TEST(SchemaValues, ParsingAndFormattingGoThroughTheDeclaredType) {
    const Schema s = shop_schema();
    EXPECT_EQ(std::get<std::int64_t>(s.parse("price", "4000")), 4000);
    EXPECT_EQ(s.format("created_at", s.parse("created_at", "2026-08-13")),
              "2026-08-13T00:00:00Z");
    EXPECT_THROW(s.parse("price", "abc"), std::invalid_argument);
}

TEST(SchemaPersistence, ASchemaSurvivesASaveAndLoad) {
    // Without this a stored plan is uninterpretable: "encoding: composite"
    // says nothing unless the key type is known.
    const Schema s = shop_schema();
    const Schema back = Schema::parse_json(s.serialize());

    ASSERT_EQ(back.size(), s.size());
    for (const ColumnDef& c : s.columns()) {
        ASSERT_TRUE(back.has(c.name));
        EXPECT_EQ(back.type_of(c.name), c.type);
        EXPECT_EQ(back.column(c.name).dim, c.dim);
    }
}

TEST(SchemaPersistence, AnEmptySchemaRoundTrips) {
    const Schema back = Schema::parse_json(Schema{}.serialize());
    EXPECT_TRUE(back.empty());
}

TEST(SchemaPersistence, UnknownFieldsAreSkippedRatherThanRejected) {
    // Same policy the index catalog takes: a file written by a later build
    // still loads here.
    const std::string blob =
        "{\"version\":1,\"note\":\"hi\",\"columns\":["
        "{\"name\":\"price\",\"type\":\"int64\",\"dim\":0,\"extra\":7}]}";
    const Schema back = Schema::parse_json(blob);
    ASSERT_EQ(back.size(), 1u);
    EXPECT_EQ(back.type_of("price"), LogicalType::Int64);
}
