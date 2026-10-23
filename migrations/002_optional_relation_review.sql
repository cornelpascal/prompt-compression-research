ALTER TABLE idea_relations ALTER COLUMN reviewer_id DROP NOT NULL;
ALTER TABLE idea_relations ADD CONSTRAINT confirmed_relation_reviewer
    CHECK(NOT confirmed OR (reviewer_id IS NOT NULL AND length(trim(reviewer_id)) > 0));
