package = "rfcn"
version = "0.1-0"

source = {
   	url = "git://fanyix",
	tag = "master"
}

description = {
   summary = "torch module for R-FCN",
   detailed = [[
   PSROIPooling module
   ]],
   homepage = "",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
   ]],
   install_command = "cd build && $(MAKE) install"
}
