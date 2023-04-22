RUST_SRC := $(shell find src -name '*.rs')

.PHONY: format
format:
	cargo fmt

.PHONY: test
test:
	$(MAKE) format
	cargo clippy --fix --allow-dirty --allow-no-vcs
	env RUST_BACKTRACE=1 cargo test

coverage/index.html: $(RUST_SRC)
	$(RM) -r coverage
	$(RM) *.profraw
	env RUSTFLAGS="-Zinstrument-coverage" cargo build
	env RUSTFLAGS="-Zinstrument-coverage" LLVM_PROFILE_FILE="%p-%m.profraw" cargo test
	grcov . --binary-path ./target/debug/ -s . -t html --branch --ignore-not-existing -o ./coverage/
	$(RM) *.profraw

.PHONY: test-coverage
test-coverage: coverage/index.html


.PHONY: presubmit
presubmit:
	$(MAKE) clean
	cargo clippy --tests -- -D warnings
	$(MAKE) test

.PHONY: clean
clean:
	cargo clean
	$(RM) Cargo.lock
	$(RM) -r coverage
	$(RM) *.profraw
	$(RM) -r target
