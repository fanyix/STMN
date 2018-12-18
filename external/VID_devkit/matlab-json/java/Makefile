# Build the org.json package.

CLASSPATH := .:junit-4.12.jar:hamcrest-core-1.3.jar
SOURCES := $(wildcard org/json/*.java)
TARGETS = $(patsubst %.java,%.class,$(SOURCES))

all: json.jar

json.jar: $(TARGETS)
	jar cvf $@ org
	jar i $@

%.class:%.java
	javac -cp $(CLASSPATH) $<

test: json.jar
	java -cp $(CLASSPATH):json.jar org.junit.runner.JUnitCore org.json.Test

clean:
	rm org/json/*.class json.jar
