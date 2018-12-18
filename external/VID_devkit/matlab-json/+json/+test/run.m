function run
%RUN Run all tests.
  tests = { ...
    @json.test.dump, ...
    @json.test.load ...
  };

  json.startup();
  for i = 1:numel(tests)
    fprintf('=== %s ===\n', func2str(tests{i}));
    tests{i}();
  end
end

