package tsml.classifiers.distance_based.utils.logging;

import java.io.OutputStream;
import java.io.PrintStream;
import java.time.Duration;
import java.util.List;
import java.util.logging.*;
import tsml.classifiers.distance_based.proximity.TimeContracter;
import tsml.classifiers.distance_based.utils.strings.StrUtils;

/**
 * Purpose: build loggers / handy logging functions
 * <p>
 * Contributors: goastler
 */
public class LogUtils {

    private LogUtils() {
    }

    public static Logger buildLogger(Object object) {
        String name;
        if(object instanceof Class) {
            name = ((Class) object).getSimpleName();
        } else if(object instanceof String) {
            name = (String) object;
        } else {
            name = object.getClass().getSimpleName() + "_" + object.hashCode();
        }
        Logger logger = Logger.getLogger(name);
        Handler[] handlers = logger.getHandlers();
        for(Handler handler : handlers) {
            logger.removeHandler(handler);
        }
        logger.setLevel(Level.SEVERE); // disable all but severe error logs by default
        logger.addHandler(buildStdOutStreamHandler(new CustomLogFormat()));
        logger.addHandler(buildStdErrStreamHandler(new CustomLogFormat()));
        logger.setUseParentHandlers(false);
        return logger;
    }

    public static class CustomLogFormat extends Formatter {

        @Override
        public String format(final LogRecord logRecord) {
            String separator = " | ";
            return logRecord.getSequenceNumber() + separator +
                logRecord.getLevel() + separator +
                logRecord.getLoggerName() + separator +
                logRecord.getSourceClassName() + separator +
                logRecord.getSourceMethodName() + System.lineSeparator() +
                logRecord.getMessage() + System.lineSeparator();
        }
    }

    public static class CustomStreamHandler extends StreamHandler {

        public CustomStreamHandler(final OutputStream out, final Formatter formatter) {
            super(out, formatter);
        }

        @Override
        public synchronized void publish(LogRecord record) {
            super.publish(record);
            flush();
        }

        @Override
        public synchronized void close() throws SecurityException {
            flush();
        }
    }

    public static StreamHandler buildStdErrStreamHandler(Formatter formatter) {
        StreamHandler soh = new CustomStreamHandler(System.err, formatter);
        soh.setLevel(Level.SEVERE); //Default StdErr Setting
        return soh;
    }

    public static StreamHandler buildStdOutStreamHandler(Formatter formatter) {
        StreamHandler soh = new CustomStreamHandler(System.out, formatter);
        soh.setLevel(Level.ALL); //Default StdOut Setting
        return soh;
    }

    public static void logTimeContract(TimeContracter timeContracter, Logger logger, String name) {
        if(timeContracter.hasTimeLimit()) {
            logger.info(() -> {
                Duration limit = Duration.ofNanos(timeContracter.getTimeLimit());
                Duration time = Duration.ofNanos(timeContracter.getTimer().getTimeNanos());
                Duration diff = limit.minus(time);
                return StrUtils.durationToHmsString(time) + " elapsed of " + StrUtils.durationToHmsString(limit) +
                    " " + name + " "
                    + "time "
                    + "limit" + System.lineSeparator() + StrUtils.durationToHmsString(diff) + " train time remaining";
            });
        }
    }

}
